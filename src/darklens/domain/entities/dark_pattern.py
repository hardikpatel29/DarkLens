"""
Core domain entities for a detected dark pattern.

Design note (read before you add fields):
    This module has ZERO third-party dependencies beyond the stdlib and
    pydantic for validation. It must never import playwright, fastapi,
    torch, transformers, or anything from `infrastructure`. If a detector
    (rule engine, NLP classifier, CV model) needs to hand you a result,
    it maps its own internal representation onto these entities at the
    boundary — these entities do not know detectors exist.

    This is what lets us add the NLP classifier in Phase 2 and a CV
    detector in Phase 4 without touching a single line in this file.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from darklens.domain.cv.labels import VisualPatternLabel
from darklens.domain.nlp.labels import NlpLabel


class PatternCategory(str, Enum):
    """
    Taxonomy of dark pattern categories.

    This follows the widely-cited academic taxonomy (Mathur et al. 2019,
    "Dark Patterns at Scale", and Gray et al. 2018 for the UX framing)
    rather than an invented one. Using an established taxonomy means your
    labels are defensible in a report or an interview, instead of "I made
    these category names up."
    """

    SNEAKING = "sneaking"                      # e.g. sneak into basket, hidden costs
    URGENCY = "urgency"                         # fake countdowns, "only X left"
    MISDIRECTION = "misdirection"                # visual interference, confirmshaming
    SOCIAL_PROOF = "social_proof"                # fake purchase notifications, fake reviews
    SCARCITY = "scarcity"                       # "low stock" claims
    OBSTRUCTION = "obstruction"                   # hard to cancel / unsubscribe
    FORCED_ACTION = "forced_action"                # forced registration, forced continuity
    PRIVACY_ZUCKERING = "privacy_zuckering"       # tricking users into sharing more data


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceSourceType(str, Enum):
    """Which detector produced a piece of evidence. Used by the fusion engine."""

    RULE_ENGINE = "rule_engine"
    NLP_CLASSIFIER = "nlp_classifier"
    CV_DETECTOR = "cv_detector"
    OCR = "ocr"


class Evidence(BaseModel):
    """
    A single, atomic, citable piece of evidence supporting a pattern
    detection. Every field here exists because "Dark Pattern Detected"
    with no evidence is not an acceptable output for this system.
    """

    source: EvidenceSourceType
    description: str = Field(..., description="Human-readable explanation of what was found")
    dom_selector: str | None = Field(
        default=None, description="CSS selector locating the offending DOM node, if applicable"
    )
    matched_rule_id: str | None = Field(
        default=None, description="Rule engine rule ID, if this evidence came from a rule"
    )
    nlp_label: NlpLabel | None = Field(
        default=None,
        description=(
            "Predicted label, if this evidence came from the NLP classifier. "
            "Deliberately a separate field from matched_rule_id rather than "
            "reusing it as a generic 'detector key' — matched_rule_id names a "
            "specific rule ID from a fixed, human-curated list; nlp_label names "
            "a class index a model predicted with some confidence. Conflating "
            "'we deterministically checked X' with 'a model thinks this is X' "
            "into one field would hide that distinction from anyone reading a "
            "report later, which matters for an explainability engine."
        ),
    )
    cv_label: VisualPatternLabel | None = Field(
        default=None,
        description="Predicted visual pattern label, if this evidence came from the CV detector.",
    )
    extracted_text: str | None = Field(
        default=None, description="Raw text this evidence was derived from (DOM text or OCR output)"
    )
    bounding_box: tuple[int, int, int, int] | None = Field(
        default=None, description="(x, y, width, height) in page pixels, for visual evidence"
    )
    raw_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="This source's own confidence, pre-fusion"
    )


class DarkPatternFinding(BaseModel):
    """
    A fused, explainable finding: the final output unit of the whole
    system. This is what the report engine renders.
    """

    id: str
    category: PatternCategory
    title: str
    severity: Severity
    confidence: float = Field(..., ge=0.0, le=1.0, description="Post-fusion calibrated confidence")
    evidence: list[Evidence]
    reasoning: str = Field(
        ..., description="Plain-language explanation of WHY this was flagged, referencing the evidence"
    )
    suggested_redesign: str | None = None

    def evidence_sources(self) -> set[EvidenceSourceType]:
        return {e.source for e in self.evidence}


class ScanResult(BaseModel):
    """Top-level output of a single website scan."""

    url: str
    scanned_at: datetime
    trust_score: float = Field(..., ge=0.0, le=100.0)
    risk_score: float = Field(..., ge=0.0, le=100.0)
    findings: list[DarkPatternFinding]
    screenshot_path: str | None = None
    scan_duration_ms: int
