"""
A Rule is pure domain logic: given a PageSnapshot, does it match, and why?

Rules live in `domain`, not `infrastructure`, deliberately. A rule like
"a checkbox is checked by default and its label implies opt-out consent"
is a business concept, not a technical concern — it has no dependency on
Playwright, FastAPI, or anything else. The RuleEngine that RUNS rules
(iterates them, wraps results as Evidence, handles a rule throwing) is
infrastructure, because "how we execute a batch of checks" is a technical
concern. Keep that distinction in mind if you add new rules: the rule
class itself should never need to import anything outside `domain`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from darklens.domain.entities.page_snapshot import PageSnapshot


class RuleMatch(BaseModel):
    """Result of a single rule evaluating a single page."""

    rule_id: str
    matched: bool
    description: str = ""
    dom_selector: str | None = None
    extracted_text: str | None = None
    confidence: float = 1.0  # deterministic rules are usually 1.0 or 0.0


class Rule(ABC):
    """One deterministic, explainable check against a PageSnapshot."""

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Stable ID, e.g. 'PRECHECKED_OPTOUT_CHECKBOX'."""

    @property
    @abstractmethod
    def category(self) -> str:
        """PatternCategory value this rule maps to."""

    @abstractmethod
    def evaluate(self, snapshot: PageSnapshot) -> list[RuleMatch]:
        """
        Evaluate this rule against a snapshot. May return zero, one, or
        multiple matches (e.g. three separate prechecked checkboxes on
        one page = three RuleMatch objects, each independently cited).
        """
