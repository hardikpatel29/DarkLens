"""CLI entrypoint. `python -m darklens.interfaces.cli.main https://example.com`"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from darklens.config import settings
from darklens.container import build_pdf_renderer, build_scan_use_case
from darklens.domain.exceptions import PageCaptureError
from darklens.infrastructure.reporting.report_generator import write_pdf_report, write_report
from darklens.logging_config import configure_logging

_SYNC_FORMATS = ("json", "html", "csv")


async def _run(url: str, output_dir: Path, formats: tuple[str, ...]) -> int:
    use_case = build_scan_use_case()
    try:
        result = await use_case.execute(url)
    except PageCaptureError as exc:
        print(f"ERROR: {exc.reason}", file=sys.stderr)
        return 1

    sync_formats = tuple(f for f in formats if f in _SYNC_FORMATS)
    paths = write_report(result, output_dir, formats=sync_formats) if sync_formats else {}

    if "pdf" in formats:
        # Built lazily, only when requested — this is the one format
        # that needs a browser launch, so a plain `--formats json` run
        # never pays that cost.
        paths["pdf"] = await write_pdf_report(result, output_dir, build_pdf_renderer())

    print(f"Trust score: {result.trust_score}  |  Risk score: {result.risk_score}")
    print(f"Findings: {len(result.findings)}")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")
    return 0


def main() -> None:
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Scan a website for dark patterns.")
    parser.add_argument("url", help="URL to scan, e.g. https://example.com")
    parser.add_argument("--output", default="./data/reports", help="Output directory for report files")
    parser.add_argument(
        "--formats",
        default="json,html",
        help="Comma-separated report formats to write: json,html,csv,pdf",
    )
    args = parser.parse_args()

    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    exit_code = asyncio.run(_run(args.url, Path(args.output), formats))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
