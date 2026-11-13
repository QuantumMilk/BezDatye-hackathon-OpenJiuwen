import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from uuid import uuid4


FRAGMENT_BYTES = 512
CREDENTIAL = re.compile(
    r"""(?im)["']?\b(?:proxy[-_]authorization|authorization|[\w-]*api[-_ ]?key)\b["']?\s*[:=][^\r\n]*"""
    r"|\bbearer\s+[^\s,;]+|\bsk-[a-z0-9_-]+"
)


class History:
    def __init__(self, emit: Callable, *, detailed: bool, secret: str = ""):
        self.emit = emit
        self.detailed = detailed
        self.secret = secret
        self.run_id = uuid4().hex
        self.sequence = count(1)

    def clean(self, value: str) -> str:
        if self.secret:
            value = value.replace(self.secret, "[redacted]")
        return CREDENTIAL.sub("[redacted]", value)

    def fragment(self, value: str, *, truncated: bool = False, tail: bool = False) -> dict:
        data = self.clean(value).encode("utf-8", errors="replace")
        excerpt = data[-FRAGMENT_BYTES:] if tail else data[:FRAGMENT_BYTES]
        return {
            "text": excerpt.decode("utf-8", errors="ignore"),
            "truncated": truncated or len(data) > FRAGMENT_BYTES,
        }

    def scope(self, task_id: str | None, *, selected: bool = False) -> "AttemptHistory":
        return AttemptHistory(self, task_id, uuid4().hex if selected else None)


@dataclass
class AttemptHistory:
    history: History
    task_id: str | None
    attempt_id: str | None
    step: int = 0
    prompts: tuple[str, ...] = ()
    file_texts: set[str] = field(default_factory=set)

    @property
    def detailed(self) -> bool:
        return self.history.detailed

    def protect_file(self, data: bytes) -> None:
        if len(data) > FRAGMENT_BYTES + 4:
            return
        try:
            value = data.decode("utf-8").rstrip("\r\n")
        except UnicodeError:
            return
        if value:
            self.file_texts.update((value, json.dumps(value, ensure_ascii=False)[1:-1]))

    def protect_workspace(self, root: Path) -> None:
        for path in root.rglob("*"):
            try:
                if path.is_file() and path.stat().st_size <= FRAGMENT_BYTES + 4:
                    with path.open("rb") as source:
                        self.protect_file(source.read(FRAGMENT_BYTES + 5))
            except OSError:
                continue

    def fragment(self, value: str, *, truncated: bool = False, payload: bool = False,
                 tail: bool = False) -> dict:
        for prompt in self.prompts:
            if prompt:
                value = value.replace(prompt, "[prompt omitted]")
                value = value.replace(json.dumps(prompt, ensure_ascii=False)[1:-1], "[prompt omitted]")
        preview = self.history.fragment(value, truncated=truncated, tail=tail)
        if payload and any(
            text in preview["text"] for text in self.file_texts
        ):
            return {"omitted": True, "truncated": preview["truncated"]}
        return preview

    def record(self, event: str, *, diagnostic: bool = False, **fields) -> None:
        if not (self.detailed or diagnostic):
            return
        self.history.emit(
            event, run_id=self.history.run_id,
            task_id=self.history.clean(self.task_id) if self.task_id is not None else None,
            attempt_id=self.attempt_id, step=self.step, event_seq=next(self.history.sequence),
            **fields,
        )
