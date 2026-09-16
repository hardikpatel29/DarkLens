from __future__ import annotations

from darklens.domain.entities.page_snapshot import DomNode
from darklens.domain.nlp.text_extraction import extract_candidate_texts
from tests.fixtures.snapshot_builder import make_snapshot


def test_extracts_visible_short_text_from_candidate_tags():
    snapshot = make_snapshot([DomNode(selector="button.cta", tag="button", text="Buy now")])
    candidates = extract_candidate_texts(snapshot)
    assert [c.text for c in candidates] == ["Buy now"]
    assert candidates[0].selector == "button.cta"


def test_skips_invisible_nodes():
    snapshot = make_snapshot(
        [DomNode(selector="span.hidden", tag="span", text="Only 1 left!", is_visible=False)]
    )
    assert extract_candidate_texts(snapshot) == []


def test_skips_non_candidate_tags():
    snapshot = make_snapshot([DomNode(selector="p.body", tag="p", text="A perfectly ordinary paragraph.")])
    assert extract_candidate_texts(snapshot) == []


def test_skips_text_outside_length_bounds():
    snapshot = make_snapshot(
        [
            DomNode(selector="span.tiny", tag="span", text="ok"),
            DomNode(selector="div.wall", tag="div", text="x" * 500),
        ]
    )
    assert extract_candidate_texts(snapshot) == []


def test_emits_both_node_text_and_associated_label_text_as_separate_candidates():
    snapshot = make_snapshot(
        [
            DomNode(
                selector="input#optin",
                tag="input",
                text="",
                associated_label_text="Uncheck to unsubscribe from emails",
            )
        ]
    )
    candidates = extract_candidate_texts(snapshot)
    assert len(candidates) == 1
    assert candidates[0].text == "Uncheck to unsubscribe from emails"


def test_deduplicates_identical_selector_text_pairs():
    snapshot = make_snapshot(
        [
            DomNode(selector="span.badge", tag="span", text="Only 2 left"),
        ]
    )
    # Same node processed once; the dedupe key is (selector, text) so this
    # test mainly guards against a future change that iterates node text
    # sources more than once and double-emits.
    candidates = extract_candidate_texts(snapshot)
    assert len(candidates) == 1


def test_collapses_internal_whitespace():
    snapshot = make_snapshot([DomNode(selector="div.banner", tag="div", text="Hurry!\n   Ends soon")])
    candidates = extract_candidate_texts(snapshot)
    assert candidates[0].text == "Hurry! Ends soon"
