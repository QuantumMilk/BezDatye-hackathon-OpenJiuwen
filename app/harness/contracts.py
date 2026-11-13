from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, computed_field

RoleType = Literal["user", "assistant", "system", "tool"]

NonBlankString = Annotated[
    str, StringConstraints(strip_whitespace=True,min_length=1)
]

class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: NonBlankString
    task_type: NonBlankString
    instructions: NonBlankString
    reward: int = Field(ge=0, strict=True)
    task_dir: Path
    workspace_dir: Path
    check: str | None = None

class Usage(BaseModel):
    input_tokens: int = Field(ge=0, strict=True)
    total_output_tokens: int = Field(ge=0, strict=True)
    reasoning_tokens: int | None = Field(default=None, ge=0, strict=True)

    @computed_field
    def cost_units(self) -> int:
        return self.input_tokens + 10 * self.total_output_tokens
