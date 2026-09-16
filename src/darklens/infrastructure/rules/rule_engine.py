"""
RuleEngineDetector: the infrastructure adapter that runs domain Rules
and exposes them through the Detector port so the (future) fusion engine
can treat "deterministic rules" and "NLP classifier" and "CV detector"
identically.

Why isolate failure per-rule: a website is untrusted input. If rule #7
throws because a site has a malformed DOM attribute, that must not take
down rules #1-6 and #8-20 along with it. This is the kind of defensive
coding a "generate all the code" prompt tends to skip, and it's exactly
the kind of thing that turns into a 2am page in production.
"""
from __future__ import annotations

import logging

from darklens.application.ports.detector import Detector
from darklens.domain.entities.dark_pattern import Evidence, EvidenceSourceType
from darklens.domain.entities.page_snapshot import PageSnapshot
from darklens.domain.rules.base import Rule
from darklens.domain.rules.structural_rules import ALL_RULES

logger = logging.getLogger(__name__)


class RuleEngineDetector(Detector):
    def __init__(self, rules: list[Rule] | None = None) -> None:
        self._rules: list[Rule] = rules if rules is not None else [cls() for cls in ALL_RULES]

    @property
    def name(self) -> str:
        return "rule_engine"

    async def detect(self, snapshot: PageSnapshot) -> list[Evidence]:
        evidence: list[Evidence] = []
        for rule in self._rules:
            try:
                matches = rule.evaluate(snapshot)
            except Exception:
                # A single misbehaving rule must never take down the scan.
                logger.exception(
                    "Rule %s raised while evaluating %s; skipping this rule for this scan.",
                    rule.rule_id,
                    snapshot.url,
                )
                continue

            for match in matches:
                if not match.matched:
                    continue
                evidence.append(
                    Evidence(
                        source=EvidenceSourceType.RULE_ENGINE,
                        description=match.description,
                        dom_selector=match.dom_selector,
                        matched_rule_id=match.rule_id,
                        extracted_text=match.extracted_text,
                        raw_confidence=match.confidence,
                    )
                )
        logger.info(
            "Rule engine produced %d evidence item(s) for %s across %d rules",
            len(evidence),
            snapshot.url,
            len(self._rules),
        )
        return evidence
