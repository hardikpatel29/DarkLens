"""
Evidence Fusion Engine.

History, kept because it's the honest record of how this boundary got
found rather than guessed:
  Phase 1: rule engine only. Nothing to fuse — one rule match = one
    finding at confidence 1.0.
  Phase 2: NLP classifier added. Still no two detectors ever agreeing on
    the SAME finding, so `fuse()` used max() across a group as a
    placeholder and said so. Metadata resolution moved to
    `pattern_catalog` — see that module for why.
  Phase 3 (this version): CV detector added, and more importantly, the
    rule engine and NLP classifier can now legitimately agree — e.g. the
    rule engine flags a prechecked opt-out checkbox AND the NLP
    classifier independently flags confirmshaming copy next to it. Two
    independent, imperfect detectors agreeing should raise confidence
    more than either alone claims — that's the actual job "fusion" was
    named for, not a synonym for "group by key and take the max".

Combination policy: noisy-OR.
    combined = 1 - prod(1 - e.raw_confidence for e in evidence_group)

Why noisy-OR and not a weighted average: a weighted average of two
0.6-confidence signals stays at 0.6, which is wrong — two independent,
moderately-confident detectors corroborating each other should push
confidence UP, not leave it flat or (worse) get dragged toward zero by
a single relatively low score. Noisy-OR treats each piece of evidence as
an independent vote for "this really is happening"; the combined
probability that at least one of them is right rises with each
additional piece of evidence. This is the standard textbook combination
rule for independent evidence toward the same binary hypothesis (Pearl,
"Probabilistic Reasoning in Intelligent Systems", 1988) — not invented
for this project.

Deterministic floor: rule-engine evidence is not a probabilistic guess
— a prechecked checkbox either is or isn't prechecked, verified by
reading the DOM. So any finding with at least one RULE_ENGINE evidence
item is floored at 0.9, regardless of what noisy-OR alone would produce
from a small evidence group. This reflects that "we checked" and "a
model thinks" are different kinds of confidence — the same distinction
`Evidence.nlp_label` vs `Evidence.matched_rule_id` already draws at the
data-model level.

[USER TASK]
The 0.9 floor and the "independent evidence" assumption behind noisy-OR
are principled defaults, not calibrated numbers. Calibrating them
requires labeled scan data (a set of real scans with human-verified
ground truth on which findings are true/false positives) that doesn't
exist yet. Once `docs/nlp_experiments.md` and a CV equivalent have real
eval runs behind them, revisit this policy: check whether noisy-OR
combined confidence is actually well-calibrated against label accuracy
(e.g. a reliability diagram), and whether 0.9 is the right floor versus,
say, deriving it from the rule engine's own false-positive rate on a
labeled sample. Replace this paragraph with those findings when done.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from functools import reduce

from darklens.application.services.pattern_catalog import (
    PATTERN_CATALOG,
    cv_key,
    nlp_key,
    rule_key,
)
from darklens.domain.entities.dark_pattern import (
    DarkPatternFinding,
    Evidence,
    EvidenceSourceType,
    Severity,
)

_RULE_ENGINE_CONFIDENCE_FLOOR = 0.9


class FusionEngine:
    """Combines Evidence (from one or more detectors) into DarkPatternFindings."""

    def fuse(self, evidence: list[Evidence]) -> list[DarkPatternFinding]:
        by_pattern: dict[str, list[Evidence]] = defaultdict(list)
        for e in evidence:
            key = self._pattern_key(e)
            if key is None:
                continue  # evidence we structurally can't attribute to a pattern
            by_pattern[key].append(e)

        findings: list[DarkPatternFinding] = []
        for key, evidence_group in by_pattern.items():
            meta = PATTERN_CATALOG.get(key)
            if meta is None:
                continue  # evidence from a pattern we don't yet have catalog metadata for

            confidence = self._combine_confidence(evidence_group)
            reasoning = self._build_reasoning(meta.title, evidence_group)

            findings.append(
                DarkPatternFinding(
                    id=str(uuid.uuid4()),
                    category=meta.category,
                    title=meta.title,
                    severity=meta.severity,
                    confidence=confidence,
                    evidence=evidence_group,
                    reasoning=reasoning,
                    suggested_redesign=meta.redesign,
                )
            )
        return findings

    @staticmethod
    def _combine_confidence(evidence_group: list[Evidence]) -> float:
        # Noisy-OR: P(at least one signal is right) = 1 - P(all wrong).
        prob_all_wrong = reduce(
            lambda acc, e: acc * (1.0 - e.raw_confidence), evidence_group, 1.0
        )
        combined = 1.0 - prob_all_wrong

        has_deterministic_evidence = any(
            e.source == EvidenceSourceType.RULE_ENGINE for e in evidence_group
        )
        if has_deterministic_evidence:
            combined = max(combined, _RULE_ENGINE_CONFIDENCE_FLOOR)

        return round(min(combined, 1.0), 4)

    @staticmethod
    def _pattern_key(e: Evidence) -> str | None:
        """
        Build the catalog lookup key for one piece of evidence. This is
        the one place in the fusion engine that knows evidence sources
        exist at all — everything after this is source-agnostic.
        """
        if e.source == EvidenceSourceType.RULE_ENGINE:
            return rule_key(e.matched_rule_id) if e.matched_rule_id else None
        if e.source == EvidenceSourceType.NLP_CLASSIFIER:
            return nlp_key(e.nlp_label) if e.nlp_label else None
        if e.source == EvidenceSourceType.CV_DETECTOR:
            return cv_key(e.cv_label) if e.cv_label else None
        # OCR evidence never carries its own pattern key — OCR only ever
        # augments a CV_DETECTOR finding with extracted text (see
        # infrastructure/cv/cv_detector.py), it doesn't independently
        # claim a pattern.
        return None

    @staticmethod
    def _build_reasoning(title: str, evidence_group: list[Evidence]) -> str:
        count = len(evidence_group)
        plural = "instance" if count == 1 else "instances"
        sources = sorted({e.source.value for e in evidence_group})
        detail = evidence_group[0].description
        return (
            f"Flagged as '{title}' based on {count} {plural} detected by "
            f"{', '.join(sources)}. {detail}"
        )


def compute_trust_and_risk_scores(findings: list[DarkPatternFinding]) -> tuple[float, float]:
    """
    Simple, transparent scoring: start at perfect trust, deduct per
    finding weighted by severity and confidence, floor at 0.

    This is intentionally not a black box. A recruiter or reviewer should
    be able to read this function and understand exactly how the number
    on the report was produced — that's the whole point of an
    "explainable" framework.
    """
    severity_weight = {
        Severity.LOW: 5.0,
        Severity.MEDIUM: 12.0,
        Severity.HIGH: 20.0,
        Severity.CRITICAL: 30.0,
    }
    deduction = sum(severity_weight[f.severity] * f.confidence for f in findings)
    trust_score = max(0.0, 100.0 - deduction)
    risk_score = min(100.0, deduction)
    return round(trust_score, 1), round(risk_score, 1)
