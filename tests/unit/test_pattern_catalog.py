from __future__ import annotations

from darklens.application.services.pattern_catalog import PATTERN_CATALOG, cv_key, nlp_key
from darklens.domain.cv.labels import ALL_VISUAL_LABELS
from darklens.domain.nlp.labels import ALL_LABELS, NlpLabel


def test_every_dark_pattern_nlp_label_has_catalog_metadata():
    """
    Guards against the failure mode this taxonomy is prone to: someone
    adds a new NlpLabel (and retrains the model to predict it) but
    forgets to add its report metadata, and it silently disappears at
    the fusion step instead of erroring loudly. NORMAL is excluded on
    purpose — see pattern_catalog.py's docstring on why it has no entry.
    """
    missing = [
        label
        for label in ALL_LABELS
        if label != NlpLabel.NORMAL and nlp_key(label) not in PATTERN_CATALOG
    ]
    assert missing == []


def test_normal_label_has_no_catalog_entry():
    assert nlp_key(NlpLabel.NORMAL) not in PATTERN_CATALOG


def test_every_visual_pattern_label_has_catalog_metadata():
    """Same guard as the NLP taxonomy test, for VisualPatternLabel."""
    missing = [label for label in ALL_VISUAL_LABELS if cv_key(label) not in PATTERN_CATALOG]
    assert missing == []
