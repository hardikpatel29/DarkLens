"""
Ports (in the hexagonal-architecture sense) that the application layer
depends on. Infrastructure implements these; application never imports
infrastructure directly.

Every detector — the deterministic rule engine today, the NLP classifier
and CV detector later — implements `Detector`. The fusion engine (Phase 3)
depends only on this interface, so plugging in a new detector is adding
one class and one line of wiring in the composition root, not modifying
the fusion engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from darklens.domain.entities.dark_pattern import Evidence
from darklens.domain.entities.page_snapshot import PageSnapshot


class Detector(ABC):
    """A single evidence-producing detector."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier, used in logs and in Evidence provenance."""

    @abstractmethod
    async def detect(self, snapshot: PageSnapshot) -> list[Evidence]:
        """
        Analyze a page snapshot and return raw evidence.

        Contract: a detector NEVER returns a final DarkPatternFinding.
        It only ever returns Evidence. Deciding what evidence adds up to
        a finding, at what confidence and severity, is the fusion
        engine's job exclusively. This separation is what keeps a single
        detector simple to unit test in isolation.
        """


class BrowserGateway(ABC):
    """Abstraction over whatever renders the page and captures a snapshot."""

    @abstractmethod
    async def capture(self, url: str) -> PageSnapshot:
        """Render `url` and return a PageSnapshot."""
