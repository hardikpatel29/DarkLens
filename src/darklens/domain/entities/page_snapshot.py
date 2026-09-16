"""
PageSnapshot is the single contract between "how we captured a page" and
"how we analyze it". Every detector (rule engine now, NLP/CV later) reads
from this object and nothing else.

Why this matters architecturally: Playwright is an implementation detail.
If you swap it for Selenium or a headless-Chrome-via-CDP client next year,
every detector keeps working unchanged, because they depend on this shape,
not on Playwright's API. This is the Dependency Inversion Principle earning
its keep, not a diagram exercise.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DomNode(BaseModel):
    """A flattened, analyzable representation of one DOM element."""

    selector: str = Field(..., description="Unique CSS selector for this node")
    tag: str
    text: str = ""
    associated_label_text: str = Field(
        default="",
        description="For form controls: text of the <label> associated via wrapping or "
        "the 'for' attribute (HTMLInputElement.labels), since inputs have no text "
        "content of their own.",
    )
    attributes: dict[str, str] = Field(default_factory=dict)
    computed_style: dict[str, str] = Field(
        default_factory=dict,
        description="Subset of computed CSS we care about: visibility, opacity, "
        "position, z-index, font-size, color, background-color",
    )
    bounding_box: tuple[int, int, int, int] | None = Field(
        default=None, description="(x, y, width, height) in viewport pixels"
    )
    is_visible: bool = True


class PageSnapshot(BaseModel):
    """Everything captured from a single rendered page load."""

    url: str
    final_url: str = Field(..., description="URL after redirects")
    html: str
    dom_nodes: list[DomNode]
    screenshot_path: str
    viewport_width: int
    viewport_height: int
    console_errors: list[str] = Field(default_factory=list)
    load_time_ms: int
