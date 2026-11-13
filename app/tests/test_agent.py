import json
import sys
import tempfile
import unittest
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import httpx

from harness.agent import local_candidate, solve
from harness.budget import BudgetManager
from harness.config import DEFAULT_MODEL
from harness.contracts import Task
from harness.execution import ExecutionResult, run_isolated
from harness.history import History
from harness.llm import ModelGateway
from harness.output import collect_changes, publish
from harness.planner import Allocation
from harness.tools import WorkspaceTools


def call(name, **arguments):
    return {"tool_calls": [{
        "id": name, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}


def candidate(answer, *evidence):
    return {"content": json.dumps({"status": "answer", "answer": answer, "evidence": evidence})}


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def test_local_search_resolves_links_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kb = root / "kb"
            kb.mkdir()
            (kb / "person.md").write_text("# Ada Lovelace\nEmployer: Example Labs\n", encoding="utf-8")
            (kb / "company.md").write_text("# Example Labs\nHeadquarters: London\n", encoding="utf-8")
            task = Task(
                task_id="search_test", task_type="search", reward=1, check="exact_str",
                task_dir=root, workspace_dir=root,
                instructions="Use only the fictional local knowledge base in workspace/kb/. "
                "In which city is the headquarters of Ada Lovelace's employer?",
            )
            tools = WorkspaceTools(root, coding=False)
            self.assertEqual(local_candidate(task, tools), "London")
            (kb / "duplicate.md").write_text("# Example Labs\nHeadquarters: York\n", encoding="utf-8")
            self.assertIsNone(local_candidate(task, tools))

    async def attempt(self, kind, instruction, files, responses, *, coding=False, units=90_000,
                      execute=None, journal=None, verify_publication=False, model=DEFAULT_MODEL):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            original = directory / "workspace"
            original.mkdir()
            work = directory / "copy"
            work.mkdir()
            for name, value in files.items():
                (work / name).parent.mkdir(parents=True, exist_ok=True)
                (original / name).parent.mkdir(parents=True, exist_ok=True)
                (work / name).write_text(value, encoding="utf-8")
                (original / name).write_text(value, encoding="utf-8")
            task = Task(
                task_id=kind, task_type=kind, instructions=instruction, reward=1,
                task_dir=directory, workspace_dir=original,
                check="unittest" if coding else "exact_str" if kind == "search" else "exact_int",
            )
            replies = iter(responses)
            requests = []

            def respond(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json={
                    "choices": [{"message": next(replies), "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 500},
                })

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(respond), base_url="https://proxy.invalid/v1/"
            ) as client:
                budget = BudgetManager(100_000, monotonic() + 30)
                gateway = ModelGateway(client, budget, model)
                tools = WorkspaceTools(
                    work, coding=coding, can_execute=sys.platform == "linux" if execute is None else execute,
                )
                result = await solve(task, tools, gateway, Allocation(units, 25), journal)
                if verify_publication:
                    if result.answer:
                        publish(task, result.answer)
                        self.assertEqual((directory / "answer.txt").read_bytes(), b"DONE\n")
                        self.assertEqual((original / "mylib.py").read_bytes(), b"value = 42\n")
                    else:
                        self.assertFalse((directory / "answer.txt").exists())
            return result, requests

    async def test_search_executes_model_query_or_falls_back(self):
        query = {"content": '{"entity":"Ada Lovelace","fields":["Works at","HQ"]}'}
        files = {
            "kb/person.md": "# Ada Lovelace\nWorks at: Example Labs\n",
            "kb/company.md": "# Example Labs\nHQ: London\n",
        }
        question = "Where is the main office of the company that employs Ada Lovelace?"
        result, requests = await self.attempt("search", question, files, [query], model="qwen3.8-27b-noreason")
        self.assertEqual(result.answer.text, "London")
        self.assertEqual(len(requests), 1)
        self.assertNotIn("London", json.dumps(requests[0]))
        self.assertNotIn("tools", requests[0])

        files["kb/company.md"] = "# Example Labs\nHQ:\n"
        result, requests = await self.attempt(
            "search", question, files, [query, {"content": '{"status":"abstain"}'}],
            model="qwen3.8-27b-noreason",
        )
        self.assertIsNone(result.answer)
        self.assertEqual(len(requests), 2)
        self.assertIn("tools", requests[1])

    async def test_closed_card_lookup_stops_only_without_other_sources(self):
        question = (
            "Use only the fictional local knowledge base in workspace/kb/. "
            "Each fact is stored only in its named field; missing fields mean unknown. "
            "In which city is the headquarters of Ada Lovelace's employer?"
        )
        files = {
            "kb/person.md": "# Ada Lovelace\nEmployer: Example Labs\n",
            "kb/company.md": "# Example Labs\nHeadquarters:\n",
        }
        abstain = {"content": '{"status":"abstain"}'}
        result, requests = await self.attempt("search", question, files, [abstain])
        self.assertEqual(result.status, "skipped")
        self.assertEqual(requests, [])
        files["notes.txt"] = "Example Labs is headquartered in London."
        result, requests = await self.attempt("search", question, files, [
            {"content": "{}"}, call("read_file", relative_path="notes.txt"),
            candidate("London", "read_file"),
        ])
        self.assertEqual(result.answer.text, "London")
        self.assertIn("tools", requests[-1])

    async def test_csv_query_ignores_unknown_values_in_excluded_rows(self):
        plan = {"content": '{"op":"sum","columns":["units","price"],'
                           '"filters":[["region","eq","north"],["month","in",[7,8]]]}'}
        files = {"sales.csv": "region,units,price,month\nnorth,3,7,7\nsouth,4,,7\nnorth,2,8,8\nnorth,6,9,9\n"}
        question = "Using workspace/sales.csv, how much revenue came from the north in months 7 and 8?"
        result, requests = await self.attempt(
            "data_analysis", question, files, [plan], model="qwen3.8-27b-noreason",
        )
        self.assertEqual(result.answer.text, "37")
        self.assertEqual(len(requests), 1)
        self.assertNotIn("north,3,7", json.dumps(requests[0]))

        files["sales.csv"] = files["sales.csv"].replace("north,3,7", "north,3,")
        result, requests = await self.attempt(
            "data_analysis", question, files, [plan, {"content": '{"status":"abstain"}'}],
            model="qwen3.8-27b-noreason",
        )
        self.assertIsNone(result.answer)
        self.assertIn("tools", requests[-1])

    async def test_math_expression_is_computed_in_one_call(self):
        result, requests = await self.attempt(
            "math", "A shop packs 23 items plus 8 more into crates of 5. How many full crates?",
            {}, [{"content": '{"expression":"(23+8)//5"}'}, {"content": '{"status":"abstain"}'}],
            model="qwen3.8-27b-noreason",
        )
        self.assertIsNotNone(result.answer)
        self.assertEqual(result.answer.text, "6")
        self.assertEqual(len(requests), 1)

    async def test_full_warehouse_sequence(self):
        prefix = "A warehouse starts with 5 boxes. "
        split = "Then the stock is split equally into 2 warehouses and it keeps one share (rounded down)."
        double = "Then the stock is doubled by merging with a sister warehouse."
        question = " How many boxes does it have at the end?"
        result, requests = await self.attempt("math", prefix + split + " " + double + question, {}, [])
        self.assertEqual((result.answer.text, requests), ("4", []))
        for operations in (split.replace("into 2", "into 0"), split + " It receives some boxes.",
                           double.removesuffix(".")):
            with self.subTest(operations=operations):
                result, requests = await self.attempt(
                    "math", prefix + operations + question, {},
                    [{"content": "{}"}, {"content": '{"status":"abstain"}'}],
                )
                self.assertEqual((result.status, len(requests)), ("skipped", 2))

    async def test_prompt_matches_tools(self):
        for coding, execute in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(coding=coding, execute=execute):
                _, requests = await self.attempt(
                    "coding" if coding else "search", "Inspect the data.", {},
                    [{"content": '{"status":"abstain"}'}], coding=coding, execute=execute,
                )
                prompt = requests[0]["messages"][0]["content"]
                for hint in ("list_files for ls", "search_text for literal grep", "read_file for cat",
                             "agent publishes answer.txt", '"evidence"', '"abstain"'):
                    self.assertIn(hint, prompt)
                self.assertEqual("run_python for python3 -c" in prompt, execute)
                self.assertEqual("run_tests to run" in prompt, execute and coding)

    async def test_search_exposes_exact_evidence_id_and_removes_write_instruction(self):
        question = "In which year was Irina Smith born?"
        instruction = (
            f"{question} When you are done, write ONLY the final answer to the file answer.txt "
            "in the current directory (for example with bash: echo 42 > answer.txt)."
        )
        read = call("read_file", relative_path="person.md")
        read["tool_calls"][0]["id"] = "chatcmpl-tool-123"
        result, requests = await self.attempt(
            "search", instruction, {"person.md": "Born: 1983\n"},
            [read, candidate("1983", "read_file person.md"), candidate("1983", "chatcmpl-tool-123")],
        )
        self.assertEqual(result.answer.text, "1983")
        self.assertEqual(json.loads(requests[0]["messages"][1]["content"])["instruction"], question)
        self.assertIn("Evidence ID: chatcmpl-tool-123", requests[1]["messages"][-1]["content"])
        self.assertIn("Copy the exact Evidence IDs", requests[2]["messages"][-1]["content"])

    async def test_data_analysis_exposes_evidence_and_rejects_wrong_reference(self):
        result, requests = await self.attempt(
            "data_analysis", "Read the number from reading.txt.", {"reading.txt": "42\n"},
            [call("read_file", relative_path="reading.txt"),
             candidate("42", "wrong"), candidate("42", "read_file")],
        )
        self.assertEqual(result.answer.text, "42")
        self.assertIn("Evidence ID: read_file", requests[1]["messages"][-1]["content"])
        self.assertIn("Copy the exact Evidence IDs", requests[2]["messages"][-1]["content"])

    async def test_spec_coding_can_abstain_after_passing_public_tests(self):
        with patch("harness.tools.run_isolated", return_value=ExecutionResult("passed")), \
             patch("harness.output.run_isolated", return_value=ExecutionResult("passed")):
            result, requests = await self.attempt(
                "coding", "Repair the implementation according to workspace/SPEC.md.",
                {"mylib.py": "value = 0\n", "test_mylib.py": "assert True\n",
                 "SPEC.md": "Require value to equal both 42 and 43.\n"},
                [call("read_file", relative_path="SPEC.md"),
                 call("write_file", relative_path="mylib.py", content="value = 42\n"),
                 call("run_tests"), {"content": '{"status":"abstain","reason":"Conflicting requirements"}'}],
                coding=True, execute=True, verify_publication=True,
            )
        self.assertEqual(result.status, "skipped")
        self.assertEqual(len(requests), 4)

    @unittest.skipUnless(sys.platform == "linux", "Linux isolation required")
    async def test_passing_public_tests_publish_without_another_request(self):
        entries = []
        history = History(lambda event, **fields: entries.append((event, fields)), detailed=True)
        tests_then_read = {"tool_calls": call("run_tests")["tool_calls"] +
                           call("read_file", relative_path="mylib.py")["tool_calls"]}
        with patch("harness.tools.run_isolated", wraps=run_isolated) as tests, \
             patch("harness.output.run_isolated", wraps=run_isolated) as repeat:
            result, requests = await self.attempt(
                "coding", "Repair the implementation.",
                {"mylib.py": "value = 0\n", "test_mylib.py": "from mylib import value\nassert value == 42\n"},
                [call("write_file", relative_path="mylib.py", content="value = 42\n"), tests_then_read],
                coding=True, verify_publication=True, journal=history.scope("coding", selected=True),
            )
        self.assertEqual((result.status, len(requests), tests.await_count, repeat.await_count), ("ready", 2, 1, 1))
        names = [event for event, _ in entries]
        ordered = ["workspace_write", "execution_finished", "validation_tests_started",
                   "validation_tests_finished", "candidate_accepted"]
        self.assertEqual([name for name in names if name in ordered], ordered)
        self.assertTrue(next(fields for event, fields in entries if event == "workspace_write")["changed"])
        skipped = next(fields for event, fields in entries if event == "tool_skipped")
        self.assertEqual((skipped["tool_call_id"], skipped["reason"]), ("read_file", "Candidate accepted"))

    async def test_rejected_test_path_reports_cause_without_publishing(self):
        for first, edit, repeat, reason in (
            ("failed", False, "passed", '"status": "failed"'),
            ("passed", False, "passed", "No coding changes"),
            ("passed", True, "failed", "Public tests failed"),
            ("passed", True, "passed", "protected file"),
        ):
            with self.subTest(reason=reason):
                entries = []
                history = History(lambda event, **fields: entries.append((event, fields)), detailed=True)
                replies = [call("run_tests"), {"content": '{"status":"abstain"}'}]
                if edit:
                    replies.insert(0, call("write_file", relative_path="mylib.py", content="value = 42\n"))

                def changes(task, work):
                    if reason == "protected file":
                        (work / "test_mylib.py").write_text("assert False\n")
                    return collect_changes(task, work)

                with patch("harness.tools.run_isolated", return_value=ExecutionResult(first)), \
                     patch("harness.output.run_isolated", return_value=ExecutionResult(repeat)), \
                     patch("harness.output.collect_changes", side_effect=changes):
                    result, requests = await self.attempt(
                        "coding", "Repair the implementation.",
                        {"mylib.py": "value = 0\n", "test_mylib.py": "assert True\n"}, replies,
                        coding=True, execute=True, verify_publication=True,
                        journal=history.scope("coding", selected=True),
                    )
                self.assertEqual(result.status, "skipped")
                self.assertIn(reason, requests[-1]["messages"][-1]["content"])
                if first == "passed":
                    self.assertTrue(any(event == "candidate_rejected" for event, _ in entries))

    async def test_base_diagnostics_distinguish_failures(self):
        entries = []
        history = History(lambda event, **fields: entries.append((event, fields)), detailed=False)
        await self.attempt(
            "search", "Find the result.", {}, [
                call("read_file", relative_path="missing.txt"), {"content": ""},
                {"content": "not JSON"}, candidate("42"),
                {"content": '{"status":"abstain"}'},
            ], journal=history.scope("search", selected=True),
        )
        self.assertEqual([event for event, _ in entries],
                         ["tool_finished", "final_json_invalid", "final_json_invalid", "candidate_rejected"])
        self.assertEqual(entries[0][1]["tool"], "read_file")
        self.assertEqual([row["empty"] for _, row in entries[1:3]], [True, False])
        self.assertIn("supporting tool", entries[3][1]["reason"]["text"])

    async def test_history_omits_sensitive_payloads_and_marks_truncation(self):
        lines = []
        history = History(
            lambda event, **fields: lines.append(json.dumps({"event": event, **fields}, ensure_ascii=False)),
            detailed=True, secret="fixture-credential",
        )
        prompt = "Read the value from the workspace."
        short_file = "fixture-file-content"
        await self.attempt(
            "search", prompt, {"input.txt": short_file}, [
                call("read_file", relative_path="input.txt"),
                {"content": short_file},
                {"content": prompt + "\nAuthorization: Bearer fixture-credential\n" + "🙂" * 300},
                candidate("42", "read_file"),
            ], journal=history.scope("search", selected=True),
        )
        rows = [json.loads(line) for line in lines]
        self.assertEqual([row["event_seq"] for row in rows], list(range(1, len(rows) + 1)))
        log = "\n".join(lines)
        for sensitive in (short_file, prompt, "Authorization", "fixture-credential"):
            self.assertNotIn(sensitive, log)
        fragments = [row["fragment"] for row in rows if row["event"] == "final_json_invalid"]
        self.assertTrue(fragments[0]["omitted"])
        self.assertTrue(fragments[1]["truncated"])
        self.assertLessEqual(len(fragments[1]["text"].encode("utf-8")), 512)
        self.assertTrue(history.fragment("short", truncated=True)["truncated"])

    async def test_math_and_data_analysis_are_local_when_supported(self):
        result, requests = await self.attempt("math", (
            "A warehouse starts with 1669 boxes. It ships 36 boxes per day for 3 days. "
            "It receives 2 deliveries of 87 boxes each. How many boxes does it have at the end?"
        ), {}, [])
        self.assertEqual((result.answer.text, requests), ("1735", []))
        result, requests = await self.attempt("data_analysis", (
            "The file sales.csv has columns order_id,region,product,month,units,price (2 rows). "
            "How many units of product 'widget' were sold in months 7 through 8 inclusive?"
        ), {"sales.csv": (
            "order_id,region,product,month,units,price\n1,a,widget,7,4,1\n2,b,widget,9,6,1\n"
        )}, [])
        self.assertEqual((result.answer.text, requests), ("4", []))

    async def test_unknown_route_repairs_invalid_candidate(self):
        result, requests = await self.attempt("unfamiliar", "Read the count.", {"data.txt": "42"}, [
            call("read_file", relative_path="data.txt"),
            candidate("forty-two", "read_file"),
            candidate("42", "read_file"),
        ])
        self.assertEqual(result.answer.text, "42")
        self.assertIn("Candidate rejected", requests[-1]["messages"][-1]["content"])

    @unittest.skipUnless(sys.platform == "linux", "Linux isolation required")
    async def test_mixed_search_and_coding(self):
        result, requests = await self.attempt("search", "Find x and double it.", {"data.txt": "x=21"}, [
            call("search_text", literal_query="x="),
            call("run_python", code="print(2 * int(open('data.txt').read().split('=')[1]))"),
            candidate("42", "run_python"),
        ])
        self.assertEqual(result.answer.text, "42")
        self.assertIn("42", requests[-1]["messages"][-1]["content"])

        result, requests = await self.attempt(
            "coding", "Repair the implementation.", {
                "mylib.py": "value = 0\n",
                "test_mylib.py": "from mylib import value\nassert value == 42\n",
            }, [
                call("write_file", relative_path="mylib.py", content="value = 1\n"),
                candidate("DONE"),
                call("write_file", relative_path="mylib.py", content="value = 42\n"),
                candidate("DONE"),
            ], coding=True,
        )
        self.assertEqual(result.answer.changes[0].after, b"value = 42\n")
        self.assertIn("Public tests failed", requests[2]["messages"][-1]["content"])

    async def test_missing_input_and_repeated_actions(self):
        result, requests = await self.attempt(
            "search", "The directory kb/ contains the records.", {}, [],
        )
        self.assertIn("missing", result.reason)
        self.assertEqual(requests, [])
        result, requests = await self.attempt(
            "search", "Find a count.", {"data.txt": "42"},
            [call("read_file", relative_path="data.txt")] * 6,
            units=40_000,
        )
        self.assertEqual(result.reason, "Repeated actions without progress")
        self.assertEqual(len(requests), 3)
