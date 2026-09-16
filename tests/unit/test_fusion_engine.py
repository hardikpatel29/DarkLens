from __future__ import annotations

from darklens.application.services.fusion_engine import (
    FusionEngine,
    compute_trust_and_risk_scores,
)
from darklens.domain.cv.labels import VisualPatternLabel
from darklens.domain.entities.dark_pattern import Evidence, EvidenceSourceType
from darklens.domain.nlp.labels import NlpLabel


def _evidence(rule_id: str, confidence: float = 1.0) -> Evidence:
    return Evidence(
        source=EvidenceSourceType.RULE_ENGINE,
        description="test evidence",
        matched_rule_id=rule_id,
        raw_confidence=confidence,
    )


def _nlp_evidence(label: NlpLabel, confidence: float = 0.9) -> Evidence:
    return Evidence(
        source=EvidenceSourceType.NLP_CLASSIFIER,
        description="test nlp evidence",
        nlp_label=label,
        raw_confidence=confidence,
    )


def test_fuse_groups_evidence_by_rule_into_one_finding():
    engine = FusionEngine()
    evidence = [
        _evidence("PRECHECKED_OPTOUT_CHECKBOX"),
        _evidence("PRECHECKED_OPTOUT_CHECKBOX"),  # two checkboxes on the page
        _evidence("HIDDEN_UNSUBSCRIBE_LINK"),
    ]
    findings = engine.fuse(evidence)
    assert len(findings) == 2
    checkbox_finding = next(f for f in findings if f.title.startswith("Pre-checked"))
    assert len(checkbox_finding.evidence) == 2


def test_fuse_combines_multiple_rule_hits_with_noisy_or_and_floor():
    engine = FusionEngine()
    findings = engine.fuse(
        [
            _evidence("PRECHECKED_OPTOUT_CHECKBOX", confidence=1.0),
            _evidence("PRECHECKED_OPTOUT_CHECKBOX", confidence=1.0),
        ]
    )
    checkbox_finding = next(f for f in findings if f.title.startswith("Pre-checked"))
    # Deterministic rule evidence is floored at 0.9 (see fusion_engine.py) —
    # noisy-OR of two confidence-1.0 signals would already exceed that, but
    # the floor is what guarantees it regardless of individual confidences.
    assert checkbox_finding.confidence >= 0.9


def test_fuse_ignores_evidence_with_unknown_rule_id():
    engine = FusionEngine()
    findings = engine.fuse([_evidence("SOME_RULE_WITH_NO_METADATA_YET")])
    assert findings == []


def test_trust_and_risk_scores_are_complementary_bounds():
    engine = FusionEngine()
    findings = engine.fuse([_evidence("PRECHECKED_OPTOUT_CHECKBOX", confidence=1.0)])
    trust, risk = compute_trust_and_risk_scores(findings)
    assert 0.0 <= trust <= 100.0
    assert 0.0 <= risk <= 100.0
    assert trust == 100.0 - risk


def test_no_findings_means_perfect_trust():
    trust, risk = compute_trust_and_risk_scores([])
    assert trust == 100.0
    assert risk == 0.0


def test_fuse_resolves_nlp_evidence_to_catalog_metadata():
    engine = FusionEngine()
    findings = engine.fuse([_nlp_evidence(NlpLabel.FALSE_URGENCY)])
    assert len(findings) == 1
    assert findings[0].title == "Manipulative urgency language"
    assert findings[0].evidence[0].source == EvidenceSourceType.NLP_CLASSIFIER


def test_fuse_combines_multiple_nlp_hits_with_noisy_or():
    engine = FusionEngine()
    findings = engine.fuse(
        [_nlp_evidence(NlpLabel.SCARCITY, 0.7), _nlp_evidence(NlpLabel.SCARCITY, 0.95)]
    )
    assert len(findings) == 1
    assert len(findings[0].evidence) == 2
    # noisy-OR: 1 - (1-0.7)*(1-0.95) = 1 - 0.3*0.05 = 0.985.
    # No RULE_ENGINE evidence in this group, so no floor applies.
    assert findings[0].confidence == 0.985


def test_fuse_single_nlp_hit_confidence_is_unchanged_by_combination():
    engine = FusionEngine()
    findings = engine.fuse([_nlp_evidence(NlpLabel.SCARCITY, 0.73)])
    # noisy-OR of a single value is that value itself — sanity check that
    # combination doesn't distort the single-evidence case.
    assert findings[0].confidence == 0.73


def test_fuse_ignores_evidence_with_no_nlp_label():
    # Defensive case: NLP_CLASSIFIER-sourced evidence that somehow has no
    # label attached should be dropped, not crash the fusion pass.
    engine = FusionEngine()
    stray = Evidence(
        source=EvidenceSourceType.NLP_CLASSIFIER,
        description="malformed evidence",
        raw_confidence=0.8,
    )
    assert engine.fuse([stray]) == []


def test_fuse_handles_mixed_rule_and_nlp_evidence_in_one_scan():
    engine = FusionEngine()
    findings = engine.fuse(
        [_evidence("PRECHECKED_OPTOUT_CHECKBOX"), _nlp_evidence(NlpLabel.CONFIRMSHAMING)]
    )
    titles = {f.title for f in findings}
    assert titles == {"Pre-checked opt-out consent checkbox", "Confirmshaming language"}


def test_fuse_resolves_cv_evidence_to_catalog_metadata():
    engine = FusionEngine()
    cv_evidence = Evidence(
        source=EvidenceSourceType.CV_DETECTOR,
        description="test cv evidence",
        cv_label=VisualPatternLabel.COUNTDOWN_TIMER,
        raw_confidence=0.8,
    )
    findings = engine.fuse([cv_evidence])
    assert len(findings) == 1
    assert findings[0].title == "Countdown timer"


def test_fuse_ignores_evidence_with_no_cv_label():
    engine = FusionEngine()
    stray = Evidence(source=EvidenceSourceType.CV_DETECTOR, description="malformed", raw_confidence=0.8)
    assert engine.fuse([stray]) == []


def test_fuse_ignores_bare_ocr_evidence():
    # OCR never independently claims a pattern — see fusion_engine.py's
    # _pattern_key docstring.
    engine = FusionEngine()
    ocr_only = Evidence(source=EvidenceSourceType.OCR, description="text found", raw_confidence=0.9)
    assert engine.fuse([ocr_only]) == []
