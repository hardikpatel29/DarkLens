"""
Label taxonomy for the visual/CV detector.

Same reasoning as `domain.nlp.labels`: this is the fine-grained label set
a YOLO model is trained to predict, kept separate from the report-facing
`PatternCategory` taxonomy. The mapping between the two lives in
`application.services.pattern_catalog`, not here.

Every label here exists because it's a pattern that DOM analysis
*structurally cannot see* — a countdown timer rendered as a canvas/GIF,
a fake purchase toast that's just styled divs indistinguishable in the
DOM from a real notification, a close button made deliberately tiny.
If a pattern is reliably detectable from the DOM (e.g. a prechecked
checkbox), it belongs in the rule engine, not here — running a vision
model to answer a question the DOM already answers for free is wasted
latency and an extra source of false negatives. See the "why CV exists
at all" framing in `infrastructure/cv/cv_detector.py`.
"""
from __future__ import annotations

from enum import Enum


class VisualPatternLabel(str, Enum):
    FAKE_PURCHASE_NOTIFICATION = "fake_purchase_notification"
    COUNTDOWN_TIMER = "countdown_timer"
    DISCOUNT_BADGE = "discount_badge"
    FLOATING_OVERLAY = "floating_overlay"
    SUBSCRIPTION_POPUP = "subscription_popup"
    COOKIE_POPUP = "cookie_popup"
    TINY_CLOSE_BUTTON = "tiny_close_button"
    URGENCY_BANNER = "urgency_banner"
    SCARCITY_LABEL = "scarcity_label"


# Labels whose evidence is strengthened by reading the text inside the
# detected region (a countdown timer's actual numbers, a badge's actual
# discount claim) rather than the visual class alone. Drives whether
# CvDetector bothers calling OCR for a given detection — see
# infrastructure/cv/cv_detector.py. TINY_CLOSE_BUTTON and
# FLOATING_OVERLAY are excluded: their evidence is purely
# geometric/visual (a button's size, an overlay's z-index/opacity), and
# running OCR on them would be pure latency cost for no explanatory gain.
TEXT_BEARING_LABELS: frozenset[VisualPatternLabel] = frozenset(
    {
        VisualPatternLabel.FAKE_PURCHASE_NOTIFICATION,
        VisualPatternLabel.COUNTDOWN_TIMER,
        VisualPatternLabel.DISCOUNT_BADGE,
        VisualPatternLabel.SUBSCRIPTION_POPUP,
        VisualPatternLabel.COOKIE_POPUP,
        VisualPatternLabel.URGENCY_BANNER,
        VisualPatternLabel.SCARCITY_LABEL,
    }
)

ALL_VISUAL_LABELS: tuple[VisualPatternLabel, ...] = tuple(VisualPatternLabel)
