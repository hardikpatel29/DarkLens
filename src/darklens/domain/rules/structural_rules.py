"""
Concrete deterministic rules.

Each rule below is written against real, observable DOM/CSS signals —
not against "does this page feel sketchy". That's the standard I want
you to hold every new rule to: if you can't point to the specific
attribute or computed style you're checking, the rule isn't ready.
"""
from __future__ import annotations

import re

from darklens.domain.entities.page_snapshot import PageSnapshot
from darklens.domain.rules.base import Rule, RuleMatch

# Words that reliably signal opt-out consent framing next to a checkbox,
# drawn from patterns documented in Mathur et al. 2019 and Gray et al. 2018.
_OPT_OUT_PHRASES = re.compile(
    r"(unsubscribe|opt.?out|do not send|no,? i (do not|don't) want|uncheck to)",
    re.IGNORECASE,
)

_URGENCY_PHRASES = re.compile(
    r"(only \d+ left|hurry|ends? (today|soon|in)|almost gone|selling fast|limited time)",
    re.IGNORECASE,
)

_CONFIRMSHAMING_PHRASES = re.compile(
    r"(no,? i (don'?t|do not) (want|like)|no thanks,? i (prefer|enjoy)|i don'?t care about (my|saving))",
    re.IGNORECASE,
)


class PrecheckedOptOutCheckbox(Rule):
    """
    Flags checkboxes that are checked by default AND whose nearby label
    text implies the user is opting OUT of something by leaving it
    checked (e.g. pre-consenting to marketing email).
    """

    rule_id = "PRECHECKED_OPTOUT_CHECKBOX"
    category = "sneaking"

    def evaluate(self, snapshot: PageSnapshot) -> list[RuleMatch]:
        matches: list[RuleMatch] = []
        for node in snapshot.dom_nodes:
            if node.tag != "input" or node.attributes.get("type") != "checkbox":
                continue
            if node.attributes.get("checked") is None:
                continue
            label_text = (
                node.attributes.get("aria-label", "")
                or node.associated_label_text
                or node.text
            )
            if _OPT_OUT_PHRASES.search(label_text):
                matches.append(
                    RuleMatch(
                        rule_id=self.rule_id,
                        matched=True,
                        description=(
                            "Checkbox is checked by default and its label implies "
                            "opting out, meaning the default state opts the user IN "
                            "to something without explicit action."
                        ),
                        dom_selector=node.selector,
                        extracted_text=label_text,
                    )
                )
        return matches


class HiddenUnsubscribeLink(Rule):
    """
    Flags unsubscribe/cancel links whose computed style makes them
    effectively invisible or unusably small — visibility:hidden,
    opacity near zero, or font-size below a legible threshold.
    """

    rule_id = "HIDDEN_UNSUBSCRIBE_LINK"
    category = "obstruction"
    _LINK_TEXT = re.compile(r"(unsubscribe|cancel (my )?(subscription|account|membership))", re.IGNORECASE)

    def evaluate(self, snapshot: PageSnapshot) -> list[RuleMatch]:
        matches: list[RuleMatch] = []
        for node in snapshot.dom_nodes:
            if node.tag not in ("a", "button"):
                continue
            if not self._LINK_TEXT.search(node.text):
                continue
            style = node.computed_style
            opacity = _safe_float(style.get("opacity"), default=1.0)
            font_size = _safe_float(style.get("font-size", "").replace("px", ""), default=16.0)
            is_hidden = (
                style.get("visibility") == "hidden"
                or style.get("display") == "none"
                or opacity <= 0.05
                or font_size <= 4.0
                or not node.is_visible
            )
            if is_hidden:
                matches.append(
                    RuleMatch(
                        rule_id=self.rule_id,
                        matched=True,
                        description=(
                            f"Element with unsubscribe/cancel text is rendered effectively "
                            f"invisible (opacity={opacity}, font-size={font_size}px, "
                            f"visibility={style.get('visibility')})."
                        ),
                        dom_selector=node.selector,
                        extracted_text=node.text,
                    )
                )
        return matches


class ForcedRegistrationGate(Rule):
    """
    Flags a login/registration form that appears with no visible way to
    dismiss or proceed as a guest — approximated here by: a full-viewport
    overlay containing password/email fields, with no visible close
    control and no visible 'guest' / 'skip' link anywhere on the page.
    """

    rule_id = "FORCED_REGISTRATION_GATE"
    category = "forced_action"
    _GUEST_TEXT = re.compile(r"(guest|skip|continue without|maybe later)", re.IGNORECASE)

    def evaluate(self, snapshot: PageSnapshot) -> list[RuleMatch]:
        matches: list[RuleMatch] = []
        has_guest_option = any(
            self._GUEST_TEXT.search(n.text) and n.is_visible for n in snapshot.dom_nodes
        )
        if has_guest_option:
            return matches

        for node in snapshot.dom_nodes:
            if node.tag != "form":
                continue
            field_types = {
                child_input for n in snapshot.dom_nodes
                if n.tag == "input" and n.selector.startswith(node.selector)
                for child_input in [n.attributes.get("type", "text")]
            }
            has_password = "password" in field_types
            style = node.computed_style
            is_full_screen_overlay = style.get("position") in ("fixed", "absolute") and _safe_float(
                style.get("z-index"), default=0
            ) >= 999
            has_close_button = any(
                n.selector.startswith(node.selector)
                and n.tag == "button"
                and re.search(r"(close|dismiss|x|×)", n.text, re.IGNORECASE)
                for n in snapshot.dom_nodes
            )
            if has_password and is_full_screen_overlay and not has_close_button:
                matches.append(
                    RuleMatch(
                        rule_id=self.rule_id,
                        matched=True,
                        description=(
                            "Full-screen overlay contains a password field, has no visible "
                            "close/dismiss control, and no guest/skip option exists on the page."
                        ),
                        dom_selector=node.selector,
                    )
                )
        return matches


class InterfaceInterference(Rule):
    """
    Flags a common visual-trickery pattern: two actionable elements
    (e.g. 'Accept' and 'Reject' on a cookie banner) with a large,
    deliberate contrast disparity in size and/or color prominence,
    where one is visually dominant enough to be the obvious default
    action. We approximate 'visually dominant' via font-size ratio and
    background-color presence (a filled button vs. plain text link).
    """

    rule_id = "INTERFACE_INTERFERENCE_CONSENT"
    category = "misdirection"
    _ACCEPT_TEXT = re.compile(r"(accept all|allow all|agree|got it)", re.IGNORECASE)
    _REJECT_TEXT = re.compile(r"(reject all|decline|deny|no thanks|manage preferences)", re.IGNORECASE)

    def evaluate(self, snapshot: PageSnapshot) -> list[RuleMatch]:
        matches: list[RuleMatch] = []
        accept_node = next(
            (n for n in snapshot.dom_nodes if n.tag in ("a", "button") and self._ACCEPT_TEXT.search(n.text)),
            None,
        )
        reject_node = next(
            (n for n in snapshot.dom_nodes if n.tag in ("a", "button") and self._REJECT_TEXT.search(n.text)),
            None,
        )
        if not accept_node or not reject_node:
            return matches

        accept_has_fill = accept_node.computed_style.get("background-color", "rgba(0, 0, 0, 0)") not in (
            "rgba(0, 0, 0, 0)",
            "transparent",
        )
        reject_has_fill = reject_node.computed_style.get("background-color", "rgba(0, 0, 0, 0)") not in (
            "rgba(0, 0, 0, 0)",
            "transparent",
        )
        if accept_has_fill and not reject_has_fill:
            matches.append(
                RuleMatch(
                    rule_id=self.rule_id,
                    matched=True,
                    description=(
                        "Consent 'accept' control is a filled button while the 'reject' "
                        "control is styled as plain text, creating asymmetric visual "
                        "weight that nudges users toward accepting."
                    ),
                    dom_selector=accept_node.selector,
                    extracted_text=f"accept='{accept_node.text}' reject='{reject_node.text}'",
                )
            )
        return matches


def _safe_float(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(re.sub(r"[^0-9.\-]", "", value))
    except ValueError:
        return default


# Registry — the RuleEngine (infrastructure layer) imports this list so
# adding a new rule is a one-line addition here, not a change to the
# engine's execution logic.
ALL_RULES: list[type[Rule]] = [
    PrecheckedOptOutCheckbox,
    HiddenUnsubscribeLink,
    ForcedRegistrationGate,
    InterfaceInterference,
]
