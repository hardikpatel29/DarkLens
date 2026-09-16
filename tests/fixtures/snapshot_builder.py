"""
Test helper for constructing PageSnapshot objects by hand.

This is the payoff of the PageSnapshot abstraction from a testing
perspective: rule unit tests never launch a browser. They build a
snapshot directly and assert on rule output — fast, deterministic,
no network flakiness in CI.
"""
from __future__ import annotations

from darklens.domain.entities.page_snapshot import DomNode, PageSnapshot


def make_snapshot(nodes: list[DomNode], url: str = "https://example-test.com") -> PageSnapshot:
    return PageSnapshot(
        url=url,
        final_url=url,
        html="<html></html>",
        dom_nodes=nodes,
        screenshot_path="/tmp/fake.png",
        viewport_width=1440,
        viewport_height=900,
        load_time_ms=100,
    )


def checkbox_node(
    selector: str = "#marketing-optin",
    checked: bool = True,
    label_text: str = "Uncheck to unsubscribe from our newsletter",
) -> DomNode:
    attrs = {"type": "checkbox"}
    if checked:
        attrs["checked"] = "checked"
    # Real inputs have no text content; label text comes from the
    # associated <label> element (see PlaywrightBrowserGateway extraction
    # script, which uses HTMLInputElement.labels). Mirror that here so
    # this fixture doesn't mask the bug that caused test_end_to_end_scan
    # to fail against a real browser.
    return DomNode(selector=selector, tag="input", associated_label_text=label_text, attributes=attrs)


def link_node(
    selector: str = "a.unsubscribe",
    text: str = "Unsubscribe",
    style: dict[str, str] | None = None,
    is_visible: bool = True,
) -> DomNode:
    return DomNode(
        selector=selector,
        tag="a",
        text=text,
        computed_style=style or {},
        is_visible=is_visible,
    )
