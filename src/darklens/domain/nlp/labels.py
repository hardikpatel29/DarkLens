"""
Label taxonomy for the manipulative-UX-language classifier.

Design note (read before you add a label): this is a SEPARATE taxonomy
from `PatternCategory` in `domain.entities.dark_pattern`, on purpose.
`PatternCategory` is the coarse-grained academic taxonomy (Mathur et al.
2019 / Gray et al. 2018) that the whole report is organized around —
8 categories, stable, unlikely to change. `NlpLabel` is the fine-grained
set of *sentence-level* labels the classifier is actually trained to
predict, matched to how the Mathur / Yamana Lab dataset is labeled.

Keeping them separate means the classifier's label set can evolve
(add "Trick Questions" as a label, retrain) without touching the report
taxonomy, and the report taxonomy can be re-organized without retraining
a model. The mapping between the two lives in
`application.services.pattern_catalog`, not here — this module has no
opinion about severity, report copy, or redesign suggestions. It only
answers "what can the classifier say".

NORMAL is a first-class label, not an absence of a label. A classifier
that only ever sees dark-pattern examples during training learns nothing
about what *ordinary* UX copy looks like, and will happily call
"Free shipping on orders over $50" a false discount. Every batch fed to
this classifier includes negative (NORMAL) examples for exactly that
reason — see scripts/train_nlp_classifier.py.
"""
from __future__ import annotations

from enum import Enum


class NlpLabel(str, Enum):
    NORMAL = "normal"
    CONFIRMSHAMING = "confirmshaming"
    FALSE_URGENCY = "false_urgency"
    SCARCITY = "scarcity"
    FEAR_BASED_COPY = "fear_based_copy"
    HIDDEN_SUBSCRIPTION = "hidden_subscription"
    MISLEADING_CONSENT = "misleading_consent"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    FALSE_DISCOUNT = "false_discount"
    FORCED_CONTINUITY = "forced_continuity"


# The classifier is trained as single-label, multi-class (softmax over
# these 10 classes), not multi-label sigmoid. Reasoning: the Mathur /
# Yamana Lab dataset annotates each sentence with exactly one dominant
# manipulation strategy, not a set — a sentence like "Only 2 left, offer
# ends tonight!" is annotated as one thing, not independently as both
# Scarcity and False Urgency. Modeling it as multi-label would invent
# co-occurrence structure the data doesn't support and would need a
# calibrated per-class threshold on top, instead of a single argmax.
# If real report data later shows sentences that clearly need >1 label,
# that's a reason to revisit this — not a reason to guess now.
ALL_LABELS: tuple[NlpLabel, ...] = tuple(NlpLabel)
