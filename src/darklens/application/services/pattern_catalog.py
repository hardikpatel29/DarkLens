"""
Single source of truth for "what does this detected pattern mean to a
human reading the report": category, title, severity, redesign advice.

Refactor note (Phase 2): in Phase 1 this table lived inline in
fusion_engine.py as `_RULE_METADATA`, keyed by rule ID, because rule ID
was the only kind of pattern key that existed. The Phase 1 README
predicted "adding the NLP classifier means implementing one Detector
subclass and adding one line to container.py — nothing in application
changes." That turned out to be not quite true, and it's worth saying
so plainly instead of quietly patching around it: the NLP classifier
produces evidence keyed by a *label*, not a rule ID, and
`FusionEngine.fuse()` had rule-ID grouping baked into its control flow.
The honest fix is this module — pull metadata lookup out of the fusion
engine entirely, key it by a detector-agnostic string, and let
`fuse()` go back to knowing nothing about rules OR labels specifically.
That's a better boundary than the one Phase 1 guessed at, found by
actually hitting the case it was supposed to handle — which is the
normal way architecture boundaries get proven, not a Phase 1 mistake.

Key format: "{source}:{id}", e.g. "rule_engine:PRECHECKED_OPTOUT_CHECKBOX"
or "nlp_classifier:false_urgency". Building this key is the fusion
engine's job (it has the Evidence); resolving it to metadata is this
module's job.
"""
from __future__ import annotations

from pydantic import BaseModel

from darklens.domain.cv.labels import VisualPatternLabel
from darklens.domain.entities.dark_pattern import EvidenceSourceType, PatternCategory, Severity
from darklens.domain.nlp.labels import NlpLabel


class PatternMetadata(BaseModel):
    category: PatternCategory
    title: str
    severity: Severity
    redesign: str


def rule_key(rule_id: str) -> str:
    return f"{EvidenceSourceType.RULE_ENGINE.value}:{rule_id}"


def nlp_key(label: NlpLabel) -> str:
    return f"{EvidenceSourceType.NLP_CLASSIFIER.value}:{label.value}"


def cv_key(label: VisualPatternLabel) -> str:
    return f"{EvidenceSourceType.CV_DETECTOR.value}:{label.value}"


# --- Rule engine patterns (Phase 1) -----------------------------------
_RULE_PATTERNS: dict[str, PatternMetadata] = {
    rule_key("PRECHECKED_OPTOUT_CHECKBOX"): PatternMetadata(
        category=PatternCategory.SNEAKING,
        title="Pre-checked opt-out consent checkbox",
        severity=Severity.HIGH,
        redesign="Default the checkbox to unchecked and require an explicit opt-in action.",
    ),
    rule_key("HIDDEN_UNSUBSCRIBE_LINK"): PatternMetadata(
        category=PatternCategory.OBSTRUCTION,
        title="Unsubscribe/cancel control is effectively hidden",
        severity=Severity.HIGH,
        redesign=(
            "Render the unsubscribe/cancel control at the same visual "
            "prominence as other primary actions."
        ),
    ),
    rule_key("FORCED_REGISTRATION_GATE"): PatternMetadata(
        category=PatternCategory.FORCED_ACTION,
        title="Forced account registration with no guest path",
        severity=Severity.MEDIUM,
        redesign=(
            "Offer a visible 'continue as guest' or dismiss option "
            "alongside the registration prompt."
        ),
    ),
    rule_key("INTERFACE_INTERFERENCE_CONSENT"): PatternMetadata(
        category=PatternCategory.MISDIRECTION,
        title="Asymmetric visual weighting on consent choices",
        severity=Severity.MEDIUM,
        redesign="Give 'Accept' and 'Reject' controls equal visual styling and prominence.",
    ),
}


# --- NLP classifier patterns (Phase 2) --------------------------------
# Maps each NlpLabel onto the report-facing PatternCategory taxonomy.
# NORMAL is intentionally absent — it means "not a dark pattern" and
# never reaches the fusion engine as evidence in the first place (see
# NlpClassifierDetector). If NORMAL evidence somehow does arrive here,
# the catalog lookup miss causes it to be silently dropped, matching
# the Phase 1 behavior for unrecognized rule IDs — a documented,
# consistent fail-safe, not a special case bolted on for this label.
#
# Mapping rationale, since "which of 8 categories" is a judgment call
# worth writing down rather than leaving implicit in a dict literal:
#   CONFIRMSHAMING          -> MISDIRECTION   (guilt-based framing of a decline option)
#   FALSE_URGENCY           -> URGENCY        (direct match)
#   SCARCITY                -> SCARCITY       (direct match)
#   FEAR_BASED_COPY         -> MISDIRECTION   (manipulative framing, not a false claim per se)
#   HIDDEN_SUBSCRIPTION     -> FORCED_ACTION  (continuity the user didn't clearly agree to)
#   MISLEADING_CONSENT      -> PRIVACY_ZUCKERING (consent language engineered to mislead)
#   EMOTIONAL_MANIPULATION  -> MISDIRECTION   (framing-based, not structural)
#   FALSE_DISCOUNT          -> MISDIRECTION   (misleading pricing claim, not concealment)
#   FORCED_CONTINUITY       -> FORCED_ACTION  (direct match, Mathur et al. terminology)
_NLP_PATTERNS: dict[str, PatternMetadata] = {
    nlp_key(NlpLabel.CONFIRMSHAMING): PatternMetadata(
        category=PatternCategory.MISDIRECTION,
        title="Confirmshaming language",
        severity=Severity.LOW,
        redesign="Phrase the decline option neutrally instead of shaming the user for choosing it.",
    ),
    nlp_key(NlpLabel.FALSE_URGENCY): PatternMetadata(
        category=PatternCategory.URGENCY,
        title="Manipulative urgency language",
        severity=Severity.MEDIUM,
        redesign="Only display countdowns/deadlines that are real and independently verifiable.",
    ),
    nlp_key(NlpLabel.SCARCITY): PatternMetadata(
        category=PatternCategory.SCARCITY,
        title="Unverified scarcity claim",
        severity=Severity.MEDIUM,
        redesign="Only show stock-level claims backed by real inventory data.",
    ),
    nlp_key(NlpLabel.FEAR_BASED_COPY): PatternMetadata(
        category=PatternCategory.MISDIRECTION,
        title="Fear-based persuasive copy",
        severity=Severity.MEDIUM,
        redesign="Replace fear-based framing with a neutral, factual statement of consequences.",
    ),
    nlp_key(NlpLabel.HIDDEN_SUBSCRIPTION): PatternMetadata(
        category=PatternCategory.FORCED_ACTION,
        title="Subscription terms disclosed unclearly",
        severity=Severity.HIGH,
        redesign="State recurring billing terms in the same place and size as the price.",
    ),
    nlp_key(NlpLabel.MISLEADING_CONSENT): PatternMetadata(
        category=PatternCategory.PRIVACY_ZUCKERING,
        title="Misleading consent language",
        severity=Severity.HIGH,
        redesign="Describe exactly what the user is consenting to, in plain language, before the action.",
    ),
    nlp_key(NlpLabel.EMOTIONAL_MANIPULATION): PatternMetadata(
        category=PatternCategory.MISDIRECTION,
        title="Emotionally manipulative copy",
        severity=Severity.LOW,
        redesign="Replace emotionally loaded phrasing with neutral, informative language.",
    ),
    nlp_key(NlpLabel.FALSE_DISCOUNT): PatternMetadata(
        category=PatternCategory.MISDIRECTION,
        title="Unverified or misleading discount claim",
        severity=Severity.MEDIUM,
        redesign="Only advertise discounts against a real, verifiable prior price.",
    ),
    nlp_key(NlpLabel.FORCED_CONTINUITY): PatternMetadata(
        category=PatternCategory.FORCED_ACTION,
        title="Forced continuity / auto-renewal framing",
        severity=Severity.HIGH,
        redesign="Send a reminder before auto-renewal and require an explicit opt-in for continuity.",
    ),
}

PATTERN_CATALOG: dict[str, PatternMetadata] = {**_RULE_PATTERNS, **_NLP_PATTERNS}


# --- CV detector patterns (Phase 4) -----------------------------------
_CV_PATTERNS: dict[str, PatternMetadata] = {
    cv_key(VisualPatternLabel.FAKE_PURCHASE_NOTIFICATION): PatternMetadata(
        category=PatternCategory.SOCIAL_PROOF,
        title="Fake purchase/activity notification popup",
        severity=Severity.MEDIUM,
        redesign="Remove simulated activity notifications, or source them from verified real events only.",
    ),
    cv_key(VisualPatternLabel.COUNTDOWN_TIMER): PatternMetadata(
        category=PatternCategory.URGENCY,
        title="Countdown timer",
        severity=Severity.MEDIUM,
        redesign="Only display a countdown tied to a real, fixed deadline that doesn't reset per visit.",
    ),
    cv_key(VisualPatternLabel.DISCOUNT_BADGE): PatternMetadata(
        category=PatternCategory.MISDIRECTION,
        title="Discount badge overlay",
        severity=Severity.LOW,
        redesign="Verify the discount badge reflects a real, recent prior price.",
    ),
    cv_key(VisualPatternLabel.FLOATING_OVERLAY): PatternMetadata(
        category=PatternCategory.OBSTRUCTION,
        title="Persistent floating overlay obstructing content",
        severity=Severity.LOW,
        redesign="Make the overlay easily dismissible and non-blocking of primary content/navigation.",
    ),
    cv_key(VisualPatternLabel.SUBSCRIPTION_POPUP): PatternMetadata(
        category=PatternCategory.FORCED_ACTION,
        title="Subscription/upsell popup",
        severity=Severity.MEDIUM,
        redesign="Ensure the popup has an equally prominent, easy-to-find dismiss option.",
    ),
    cv_key(VisualPatternLabel.COOKIE_POPUP): PatternMetadata(
        category=PatternCategory.PRIVACY_ZUCKERING,
        title="Cookie consent popup with unbalanced choice design",
        severity=Severity.MEDIUM,
        redesign="Give 'Accept' and 'Reject' equal visual weight and one-click access.",
    ),
    cv_key(VisualPatternLabel.TINY_CLOSE_BUTTON): PatternMetadata(
        category=PatternCategory.OBSTRUCTION,
        title="Deliberately tiny or low-contrast close button",
        severity=Severity.MEDIUM,
        redesign="Size the close control consistently with other interactive controls (min ~44x44px target).",
    ),
    cv_key(VisualPatternLabel.URGENCY_BANNER): PatternMetadata(
        category=PatternCategory.URGENCY,
        title="Urgency banner",
        severity=Severity.MEDIUM,
        redesign="Only display urgency messaging backed by a real, verifiable condition.",
    ),
    cv_key(VisualPatternLabel.SCARCITY_LABEL): PatternMetadata(
        category=PatternCategory.SCARCITY,
        title="Visual scarcity label",
        severity=Severity.MEDIUM,
        redesign="Only show stock-level claims backed by real inventory data.",
    ),
}

PATTERN_CATALOG.update(_CV_PATTERNS)
