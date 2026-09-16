"""
Unit tests for domain rules. Each test asserts BOTH the positive case
(pattern present -> rule fires) and a negative case (pattern absent ->
rule stays silent). A detector that only ever fires is worthless; a
false-positive-prone rule erodes trust in the whole report faster than
a missed detection does.
"""
from __future__ import annotations

from darklens.domain.entities.page_snapshot import DomNode
from darklens.domain.rules.structural_rules import (
    ForcedRegistrationGate,
    HiddenUnsubscribeLink,
    InterfaceInterference,
    PrecheckedOptOutCheckbox,
)
from tests.fixtures.snapshot_builder import checkbox_node, link_node, make_snapshot


class TestPrecheckedOptOutCheckbox:
    def test_fires_on_prechecked_optout_language(self):
        snapshot = make_snapshot([checkbox_node(checked=True, label_text="Uncheck to opt-out of emails")])
        matches = PrecheckedOptOutCheckbox().evaluate(snapshot)
        assert len(matches) == 1
        assert matches[0].rule_id == "PRECHECKED_OPTOUT_CHECKBOX"
        assert matches[0].dom_selector == "#marketing-optin"

    def test_silent_when_unchecked(self):
        snapshot = make_snapshot([checkbox_node(checked=False, label_text="Uncheck to opt-out of emails")])
        assert PrecheckedOptOutCheckbox().evaluate(snapshot) == []

    def test_silent_when_checked_but_no_optout_language(self):
        snapshot = make_snapshot([checkbox_node(checked=True, label_text="I agree to the Terms of Service")])
        assert PrecheckedOptOutCheckbox().evaluate(snapshot) == []


class TestHiddenUnsubscribeLink:
    def test_fires_when_opacity_near_zero(self):
        snapshot = make_snapshot([link_node(style={"opacity": "0.01"})])
        matches = HiddenUnsubscribeLink().evaluate(snapshot)
        assert len(matches) == 1

    def test_fires_when_visibility_hidden(self):
        snapshot = make_snapshot([link_node(style={"visibility": "hidden"})])
        assert len(HiddenUnsubscribeLink().evaluate(snapshot)) == 1

    def test_silent_when_normally_visible(self):
        snapshot = make_snapshot(
            [link_node(style={"opacity": "1", "visibility": "visible", "font-size": "16px"})]
        )
        assert HiddenUnsubscribeLink().evaluate(snapshot) == []

    def test_silent_for_unrelated_links(self):
        snapshot = make_snapshot([link_node(text="Contact us", style={"opacity": "0.01"})])
        assert HiddenUnsubscribeLink().evaluate(snapshot) == []


class TestForcedRegistrationGate:
    def test_fires_on_password_overlay_with_no_escape(self):
        nodes = [
            DomNode(
                selector="#gate-form",
                tag="form",
                computed_style={"position": "fixed", "z-index": "9999"},
            ),
            DomNode(
                selector="#gate-form input[type=password]",
                tag="input",
                attributes={"type": "password"},
            ),
        ]
        matches = ForcedRegistrationGate().evaluate(make_snapshot(nodes))
        assert len(matches) == 1

    def test_silent_when_guest_option_present(self):
        nodes = [
            DomNode(
                selector="#gate-form",
                tag="form",
                computed_style={"position": "fixed", "z-index": "9999"},
            ),
            DomNode(
                selector="#gate-form input[type=password]",
                tag="input",
                attributes={"type": "password"},
            ),
            DomNode(selector=".skip-link", tag="a", text="Continue as guest"),
        ]
        assert ForcedRegistrationGate().evaluate(make_snapshot(nodes)) == []


class TestInterfaceInterference:
    def test_fires_on_asymmetric_consent_buttons(self):
        nodes = [
            DomNode(
                selector="#accept",
                tag="button",
                text="Accept All",
                computed_style={"background-color": "rgb(0, 122, 255)"},
            ),
            DomNode(
                selector="#reject",
                tag="a",
                text="Reject All",
                computed_style={"background-color": "transparent"},
            ),
        ]
        matches = InterfaceInterference().evaluate(make_snapshot(nodes))
        assert len(matches) == 1

    def test_silent_when_visually_balanced(self):
        nodes = [
            DomNode(
                selector="#accept",
                tag="button",
                text="Accept All",
                computed_style={"background-color": "rgb(0, 122, 255)"},
            ),
            DomNode(
                selector="#reject",
                tag="button",
                text="Reject All",
                computed_style={"background-color": "rgb(200, 200, 200)"},
            ),
        ]
        assert InterfaceInterference().evaluate(make_snapshot(nodes)) == []
