#!/usr/bin/env python
"""
Prepares data/nlp/train.csv and data/nlp/val.csv for train_nlp_classifier.py.

Sources combined (in order):
  1. dark-patterns-v2.csv   — Mathur et al. 2019, already in this repo
  2. yamanalab/ec-darkpattern — HuggingFace extension (fetched if `datasets`
                                is installed; skipped gracefully if not)
  3. Curated NORMAL examples — bundled in this script (~250 examples of
                                plain e-commerce UX copy that is NOT a dark
                                pattern; required so the classifier learns
                                what ordinary copy looks like)

Output format expected by train_nlp_classifier.py:
    CSV with exactly two columns:  text, label
    where label is one of the NlpLabel values in domain/nlp/labels.py

Class balancing strategy:
    - Classes below MIN_CLASS_SIZE are oversampled (repeated) to reach target.
    - Classes above MAX_CLASS_SIZE are downsampled to avoid dominating loss.

Usage:
    python scripts/prepare_nlp_dataset.py
    python scripts/prepare_nlp_dataset.py --csv dark-patterns-v2.csv
    python scripts/prepare_nlp_dataset.py --skip-yamana   # offline mode
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("prepare_nlp_dataset")

# ---------------------------------------------------------------------------
# Label mapping: Mathur et al. "Pattern Type" -> NlpLabel value
# ---------------------------------------------------------------------------
MATHUR_TO_NLPLABEL: dict[str, str] = {
    "Countdown Timer":                  "false_urgency",
    "Limited-time Message":             "false_urgency",
    "Low-stock Message":                "scarcity",
    "High-demand Message":              "scarcity",
    "Activity Notification":            "fear_based_copy",
    "Testimonials of Uncertain Origin": "fear_based_copy",
    "Confirmshaming":                   "confirmshaming",
    "Hidden Subscription":              "hidden_subscription",
    "Hard to Cancel":                   "forced_continuity",
    "Pressured Selling":                "emotional_manipulation",
    "Visual Interference":              "misleading_consent",
    "Trick Questions":                  "misleading_consent",
    "Forced Enrollment":                "misleading_consent",
    "Sneak into Basket":                "false_discount",
    "Hidden Costs":                     "false_discount",
}

# ---------------------------------------------------------------------------
# Curated NORMAL examples (~250 plain e-commerce sentences, NOT dark patterns)
# ---------------------------------------------------------------------------
NORMAL_EXAMPLES: list[str] = [
    "Free shipping on orders over $50.",
    "Ships within 2-3 business days.",
    "Estimated delivery: Monday, August 5.",
    "Standard shipping: 5-7 business days.",
    "Express shipping: 2-3 business days.",
    "Overnight shipping available.",
    "Free in-store pickup at 3 locations.",
    "International shipping available to 50+ countries.",
    "Track your order with the link in your confirmation email.",
    "Your order has been shipped.",
    "Delivery confirmation sent to your email.",
    "Signature required on delivery.",
    "Free returns within 30 days of delivery.",
    "Easy returns — no questions asked.",
    "Return policy: 30 days for unused items.",
    "Full refund on unused, unopened items.",
    "Exchange available for a different size or color.",
    "Final sale — not eligible for return.",
    "Refund processed within 5-7 business days.",
    "Print a prepaid return label from your account page.",
    "Add to cart",
    "Add to wishlist",
    "Save for later",
    "Remove from cart",
    "Proceed to checkout",
    "Continue shopping",
    "Apply coupon code",
    "Enter promo code",
    "Order summary",
    "Subtotal: $42.99",
    "Tax: $3.87",
    "Shipping: calculated at checkout",
    "Place order",
    "Edit order",
    "View cart",
    "Your cart is empty.",
    "You have 2 items in your cart.",
    "Available in 5 colors.",
    "Comes in sizes S, M, L, XL.",
    "View size chart",
    "Product dimensions: 12 x 8 x 4 inches.",
    "Weight: 2.5 lbs.",
    "Made in USA.",
    "100% organic cotton.",
    "Machine washable. Tumble dry low.",
    "Dishwasher safe.",
    "BPA-free.",
    "Batteries included.",
    "Assembly required. Instructions included.",
    "Warranty: 1 year limited.",
    "Model number: XYZ-123.",
    "Compatible with iPhone 13, 14, and 15.",
    "Screen size: 6.1 inches.",
    "Storage: 128 GB.",
    "Battery: 4000 mAh.",
    "Camera: 50 MP wide, 12 MP ultrawide.",
    "Sort by: Price (low to high)",
    "Filter by category",
    "Showing 1-24 of 156 results.",
    "No results found. Try a different search.",
    "Narrow results by brand, size, or color.",
    "Viewing page 2 of 7.",
    "Back to category",
    "Next page",
    "Previous page",
    "View all",
    "See more",
    "Load more results",
    "Sign in to your account.",
    "Create an account.",
    "Forgot your password? Reset it here.",
    "Enter your email address.",
    "Choose a password.",
    "Remember me on this device.",
    "Log out",
    "View order history",
    "Download invoice",
    "We accept Visa, Mastercard, PayPal, and Apple Pay.",
    "Secure checkout — SSL encrypted.",
    "Your payment information is never stored on our servers.",
    "Billing address",
    "Card number",
    "Expiration date",
    "CVV / Security code",
    "Contact us",
    "Live chat available Monday-Friday, 9am-6pm EST.",
    "We typically respond within 24 hours.",
    "FAQ",
    "Help center",
    "Submit a support ticket",
    "Customer reviews",
    "4.5 out of 5 stars based on 128 reviews.",
    "Verified purchase",
    "Write a review",
    "Most helpful reviews",
    "This product has no reviews yet. Be the first.",
    "Frequently bought together",
    "Customers also viewed",
    "You might also like",
    "Similar items",
    "In stock.",
    "Usually ships in 24 hours.",
    "Available for pre-order.",
    "This item is currently out of stock.",
    "Notify me when available",
    "Subscribe to our newsletter for updates.",
    "Enter your email to receive our monthly newsletter.",
    "Unsubscribe at any time.",
    "Privacy policy",
    "Terms and conditions",
    "Accessibility statement",
    "Cookie preferences",
    "Gift wrapping available for $4.99.",
    "Add a gift message at checkout.",
    "Ship to a different address",
    "Subscribe and save 10% — cancel anytime, no fees.",
    "Delivery frequency: every 30, 60, or 90 days.",
    "Manage or pause your subscription from your account.",
    "Your subscription renews on September 1. We will email you a reminder first.",
    "Bundle: includes 3 items.",
    "Quantity: 1",
    "Color: Navy Blue",
    "Size: Medium",
    "Sold by: Example Merchant",
    "Recently viewed",
    "New arrivals",
    "Best sellers",
    "Clearance",
    "Sale — up to 40% off selected items.",
    "Price: $29.99",
    "Was $49.99. Now $34.99.",
    "Member price: $27.99. Sign in to save.",
    "Bulk discount: buy 3, get 10% off.",
    "Price match guarantee — find it cheaper, we will match it.",
    "Student discount: 15% off with valid .edu email.",
    "About us",
    "Our story",
    "Sustainability commitment",
    "Careers",
    "Press",
    "Affiliate program",
    "Sell on our marketplace",
    "Product care instructions",
    "Material: 95% polyester, 5% spandex.",
    "Country of origin: Vietnam.",
    "Max load capacity: 50 lbs.",
    "Operating temperature: 32-104 F.",
    "Input voltage: 110-240 V.",
    "Connectivity: Bluetooth 5.0, USB-C.",
    "Home / Women / Tops",
    "You are here: Electronics - Phones",
    "Edit billing address",
    "Edit shipping address",
    "Manage your account",
    "Pay with PayPal",
    "Compare up to 4 products.",
    "Share this product",
    "Copy link",
    "Saved to your wishlist.",
    "Expected availability: September 2025.",
    "You can manage your email preferences in your account.",
    "Sitemap",
    "Mark as gift",
    "Complete the look",
    "Search results for blue running shoes.",
    "Narrow results by size or brand.",
    "Fulfilled by our warehouse in Ohio.",
    "Call us: 1-800-555-0100",
    "Email: support@example.com",
    "Sort reviews: Most helpful",
    "Showing reviews 1-10.",
    "SKU: AB-12345",
    "UPC: 123456789012.",
    "Dishwasher safe on top rack only.",
    "Do not bleach.",
    "Tumble dry on low heat.",
    "Iron on low setting.",
    "Dry clean only.",
    "Hand wash recommended.",
    "Spot clean only.",
]


def _try_load_yamana() -> list[tuple[str, str]]:
    """Fetch yamanalab/ec-darkpattern from HuggingFace. Returns [] if unavailable."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        logger.warning(
            "datasets library not installed — skipping Yamana Lab. "
            "Install: pip install datasets  (or pip install -e '.[nlp]')"
        )
        return []

    try:
        logger.info("Fetching yamanalab/ec-darkpattern from HuggingFace ...")
        ds = load_dataset("yamanalab/ec-darkpattern", trust_remote_code=True)
    except Exception as exc:
        logger.warning("Could not load yamanalab/ec-darkpattern: %s — skipping.", exc)
        return []

    rows: list[dict] = []
    for split in ["train", "test", "validation", "all"]:
        if split in ds:
            rows.extend(ds[split])

    if not rows:
        logger.warning("yamanalab/ec-darkpattern loaded but no splits found.")
        return []

    sample = rows[0]
    text_col = next((c for c in ["text", "sentence", "string", "pattern_string"] if c in sample), None)
    label_col = next((c for c in ["pattern_type", "label", "category", "type"] if c in sample), None)

    if not text_col or not label_col:
        logger.warning(
            "yamanalab/ec-darkpattern has unexpected columns %s — skipping.",
            list(sample.keys()),
        )
        return []

    results: list[tuple[str, str]] = []
    skipped = 0
    for row in rows:
        text = str(row.get(text_col, "")).strip()
        src_label = str(row.get(label_col, "")).strip()
        if not text:
            continue
        nlp_label = MATHUR_TO_NLPLABEL.get(src_label)
        if nlp_label is None:
            skipped += 1
            continue
        results.append((text, nlp_label))

    logger.info("Yamana Lab: %d examples mapped, %d skipped.", len(results), skipped)
    return results


def _load_local_csv(csv_path: Path) -> list[tuple[str, str]]:
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    results: list[tuple[str, str]] = []
    skipped = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("Pattern String", "").strip()
            src_label = row.get("Pattern Type", "").strip()
            if not text:
                continue
            nlp_label = MATHUR_TO_NLPLABEL.get(src_label)
            if nlp_label is None:
                skipped += 1
                continue
            results.append((text, nlp_label))

    logger.info("Local CSV: %d mapped, %d skipped — %s", len(results), skipped, csv_path)
    return results


def _balance(examples: list[tuple[str, str]], min_size: int, max_size: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    by_label: dict[str, list] = {}
    for item in examples:
        by_label.setdefault(item[1], []).append(item)

    balanced: list[tuple[str, str]] = []
    for label, items in sorted(by_label.items()):
        n = len(items)
        if n < min_size:
            multiplier = (min_size // n) + 1
            extended = (items * multiplier)[:min_size]
            rng.shuffle(extended)
            balanced.extend(extended)
            level = logging.WARNING if n < 50 else logging.INFO
            logger.log(level, "Class '%s': %d examples — oversampled to %d.", label, n, min_size)
        elif n > max_size:
            balanced.extend(rng.sample(items, max_size))
            logger.info("Class '%s': %d examples — downsampled to %d.", label, n, max_size)
        else:
            balanced.extend(items)
            logger.info("Class '%s': %d examples — kept as-is.", label, n)

    rng.shuffle(balanced)
    return balanced


def _stratified_split(examples: list[tuple[str, str]], val_frac: float, seed: int):
    rng = random.Random(seed)
    by_label: dict[str, list] = {}
    for item in examples:
        by_label.setdefault(item[1], []).append(item)

    train, val = [], []
    for items in by_label.values():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _save_csv(examples: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(examples)
    logger.info("Saved %d rows -> %s", len(examples), path)


def _print_summary(train: list[tuple[str, str]], val: list[tuple[str, str]]) -> None:
    all_labels = sorted({label for _, label in train + val})
    tc = Counter(label for _, label in train)
    vc = Counter(label for _, label in val)
    print("\n" + "=" * 60)
    print(f"{'Label':<35} {'Train':>7} {'Val':>7} {'Total':>7}")
    print("-" * 60)
    for label in all_labels:
        t, v = tc[label], vc[label]
        print(f"{label:<35} {t:>7} {v:>7} {t + v:>7}")
    print("-" * 60)
    print(f"{'TOTAL':<35} {sum(tc.values()):>7} {sum(vc.values()):>7} {sum(tc.values()) + sum(vc.values()):>7}")
    print("=" * 60 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("dark-patterns-v2.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/nlp"))
    p.add_argument("--min-class-size", type=int, default=120)
    p.add_argument("--max-class-size", type=int, default=600)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-yamana", action="store_true", help="Offline mode: skip HuggingFace fetch")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    examples: list[tuple[str, str]] = []
    examples.extend(_load_local_csv(args.csv))

    if not args.skip_yamana:
        examples.extend(_try_load_yamana())
    else:
        logger.info("Yamana Lab skipped (--skip-yamana).")

    normal_pairs = [(t, "normal") for t in NORMAL_EXAMPLES]
    examples.extend(normal_pairs)
    logger.info("Added %d curated NORMAL examples.", len(normal_pairs))
    logger.info("Total before balancing: %d examples across all classes.", len(examples))

    balanced = _balance(examples, args.min_class_size, args.max_class_size, args.seed)
    train, val = _stratified_split(balanced, args.val_fraction, args.seed)

    _save_csv(train, args.out_dir / "train.csv")
    _save_csv(val, args.out_dir / "val.csv")
    _print_summary(train, val)

    logger.info(
        "Done. Now run:\n"
        "  pip install -e '.[nlp]'\n"
        "  python scripts/train_nlp_classifier.py \\\n"
        "      --train-csv %s/train.csv \\\n"
        "      --val-csv %s/val.csv \\\n"
        "      --output-dir models/nlp_classifier",
        args.out_dir, args.out_dir,
    )


if __name__ == "__main__":
    main()
