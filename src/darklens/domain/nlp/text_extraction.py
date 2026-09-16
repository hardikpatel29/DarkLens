"""
Candidate selection: which text on a page is worth handing to the NLP
classifier at all.

This is domain logic, not infrastructure, for the same reason a `Rule`
is domain logic (see domain/rules/base.py's module docstring): "is this
DOM text even a plausible piece of manipulative UX copy" is a business
judgment about what the product cares about, not a technical concern.
It has zero dependency on transformers, torch, or the classifier that
consumes its output.

Why this step exists at all, instead of classifying every string on the
page: a real page can have hundreds of DOM text nodes (nav items, footer
boilerplate, article body text). Running a transformer over all of them
is (a) slow — this directly trades against the project's own latency
goals — and (b) noisy, since >90% of that text is not UI microcopy and
the model was never trained to have an opinion about paragraph-length
article prose. Filtering down to short, visible, interactive-adjacent
text keeps the classifier doing the one job it was fine-tuned for.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from darklens.domain.entities.page_snapshot import DomNode, PageSnapshot

# Tags whose text is plausible UI microcopy. Deliberately excludes
# <div> and <span> — those are layout containers whose text is almost
# always inherited from a child element already in this set, causing the
# same string to be classified once per nesting level. Excluding them
# also cuts out footer boilerplate, product category headings, and
# address blocks that live in generic containers and confuse the model.
# If you add a tag here, justify it with a real false-negative case.
_CANDIDATE_TAGS = frozenset({"button", "a", "label", "input"})

# Below this, a string is almost never a complete manipulative claim
# (e.g. "OK", "x", "$"). Above this, it's very likely prose, a multi-
# clause address, or a page heading that leaked in from a container
# element rather than a single piece of UI microcopy.
_MIN_CHARS = 3
_MAX_CHARS = 120

# ---------------------------------------------------------------------------
# Blocklist: cheap pre-filter before the model ever runs.
#
# These patterns match text that is structurally impossible to be a dark
# pattern, regardless of what a classifier trained on short persuasive
# copy might say about it.  Using regex/heuristics here rather than a
# trained model keeps this step O(n) and dependency-free.
# ---------------------------------------------------------------------------

# Legal/corporate entity markers that appear in footer addresses.
_ADDRESS_MARKERS = re.compile(
    r"private limited|pvt\.?\s*ltd|llc|inc\.|gmbh|s\.a\.|"
    r"\b\d{5,6}\b|"                          # PIN / ZIP codes
    r"village|taluk|district|Karnataka|Maharashtra|Delhi|"
    r"ring road|outer ring|tech park|tech village|"
    r"buildings?\s+[A-Z][a-z]",              # "Building Alyssa, Begonia…"
    re.IGNORECASE,
)

# Navigation / UI chrome labels that appear on every page and carry no
# persuasive meaning (single words or short button rail text).
_NAV_PATTERNS = re.compile(
    r"^(login|log\s*in|sign\s*in|sign\s*up|register|cart|wishlist|"
    r"home|menu|search|close|back|next|prev|previous|skip|submit|"
    r"more|less|see\s*all|view\s*all|load\s*more|show\s*more|"
    r"account|profile|orders?|track|help|support|faq|"
    r"share|follow|like|subscribe|unsubscribe|"
    r"download|install|get\s*app|open\s*app)$",
    re.IGNORECASE,
)

# Product/content category headings — evocative but not manipulative.
_CATEGORY_PATTERNS = re.compile(
    r"\b(essentials?|collection|trending|popular|featured|"
    r"new\s+arrivals?|best\s+sellers?|top\s+picks?|"
    r"skincare|haircare|hair\s*&?\s*skin|grooming|"
    r"fashion|electronics?|appliances?|furniture|"
    r"grocery|fresh|organic)\b",
    re.IGNORECASE,
)

# Social / contact boilerplate
_SOCIAL_CONTACT = re.compile(
    r"(mail\s*us|contact\s*us|follow\s*us|connect\s*with\s*us|"
    r"social|facebook|twitter|instagram|youtube|linkedin|"
    r"@[\w.]+|[\w.]+@[\w.]+\.\w{2,})",      # social handles / email addresses
    re.IGNORECASE,
)


def _should_skip(text: str) -> bool:
    """Return True for text that is definitely not dark-pattern microcopy."""
    # Long text with address/legal markers → footer boilerplate
    if _ADDRESS_MARKERS.search(text):
        return True
    # Pure navigation labels (exact match, stripped)
    if _NAV_PATTERNS.match(text.strip()):
        return True
    # Social / contact boilerplate
    if _SOCIAL_CONTACT.search(text):
        return True
    # Category headings: short enough to pass _MAX_CHARS but not manipulative.
    # Only skip if the ENTIRE string is a category term (no surrounding copy).
    if _CATEGORY_PATTERNS.fullmatch(text.strip()):
        return True
    # Strings that are just a concatenation of navigation words (e.g.
    # "Login Login More Cart" from a repeated nav bar element).
    words = text.strip().split()
    if len(words) <= 6 and all(
        _NAV_PATTERNS.match(w) for w in words
    ):
        return True
    return False


class TextCandidate(BaseModel):
    """One piece of DOM text nominated for NLP classification."""

    selector: str
    text: str


def extract_candidate_texts(snapshot: PageSnapshot) -> list[TextCandidate]:
    """
    Walk a PageSnapshot's DOM nodes and return the deduplicated set of
    short, visible strings worth classifying.

    Deduplication key is the cleaned text alone (not selector+text).
    The same string appearing in multiple DOM elements (e.g. a button
    label repeated in a sticky header and a footer) represents one
    logical claim, not N independent ones — classifying it once and
    showing one evidence item is more accurate and avoids flooding the
    evidence list with identical lines.
    """
    seen: set[str] = set()   # keyed on text only, not (selector, text)
    candidates: list[TextCandidate] = []

    for node in snapshot.dom_nodes:
        for text in _texts_for_node(node):
            cleaned = " ".join(text.split())  # collapse whitespace/newlines
            if not (_MIN_CHARS <= len(cleaned) <= _MAX_CHARS):
                continue
            if cleaned in seen:
                continue
            if _should_skip(cleaned):
                continue
            seen.add(cleaned)
            candidates.append(TextCandidate(selector=node.selector, text=cleaned))

    return candidates


def _texts_for_node(node: DomNode) -> list[str]:
    if not node.is_visible or node.tag not in _CANDIDATE_TAGS:
        return []
    # A node can contribute both its own text AND its associated label
    # text (e.g. a checkbox's <label>) — these are genuinely different
    # strings that can each independently be manipulative, so both are
    # emitted as separate candidates rather than concatenated.
    texts = []
    if node.text.strip():
        texts.append(node.text)
    if node.associated_label_text.strip():
        texts.append(node.associated_label_text)
    return texts
