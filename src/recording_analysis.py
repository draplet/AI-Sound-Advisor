"""
src/recording_analysis.py

Recording / Broadcast Analysis Module for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt — RECORDING ANALYSIS
MODULE):
    Analyzes:
      - Audio sent to recording/stream
    Detects:
      - Vocal clarity differences
      - Loudness imbalance
      - Mix inconsistencies vs room
    Purpose:
      Improve broadcast/recording quality

Design — STRATEGY PATTERN (matching Agents 1 and 3)
---------------------------------------------------
The module compares two ``AudioMetrics`` snapshots produced by Agent 1: the
*room* reference (what the congregation hears) and the *recording* feed (what is
sent to the recorder/stream). Each difference is detected by an interchangeable
``RecordingComparator`` that emits at most one ``RecordingFinding``. The
``RecordingAnalysisEngine`` runs whatever comparators it is given and merges
their findings into a single, severity-sorted ``RecordingAnalysis``.

The default engine ships three comparators, one per spec bullet:
  - ``LoudnessImbalanceComparator``  — loudness imbalance vs the room
  - ``VocalClarityComparator``       — presence/high-frequency loss on the feed
  - ``MixInconsistencyComparator``   — low-mid tonal balance differs from room

FAILSAFE — "System must never crash during service"
----------------------------------------------------
``analyze`` wraps every comparator: a misbehaving comparator is skipped, never
fatal (mirrors ``DetectionEngine``). The engine only advises — it never controls
the mixer.

DETERMINISM
-----------
The engine holds no clock and no state; given the same two metric snapshots it
always returns the same analysis.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.audio_analysis import AudioMetrics, EnergyLevel
from src.detection import Priority

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

#: Loudness difference (LUFS) between recording and room that counts as an
#: imbalance worth flagging.
LOUDNESS_IMBALANCE_LUFS: float = 3.0

#: Loudness difference (LUFS) at/above which the imbalance is treated as HIGH.
_LOUDNESS_HIGH_LUFS: float = 6.0

#: Ordinal value of each energy level, for comparing two bands.
_ENERGY_RANK: Dict[EnergyLevel, int] = {
    EnergyLevel.LOW: 0,
    EnergyLevel.NORMAL: 1,
    EnergyLevel.HIGH: 2,
}

#: Severity ordering used to sort findings (higher = shown first).
_PRIORITY_RANK: Dict[Priority, int] = {
    Priority.CRITICAL: 3,
    Priority.HIGH: 2,
    Priority.MEDIUM: 1,
    Priority.LOW: 0,
}


def _energy_gap(room: EnergyLevel, recording: EnergyLevel) -> int:
    """Recording-minus-room band rank (negative = recording has less energy)."""
    return _ENERGY_RANK[recording] - _ENERGY_RANK[room]


# ===========================================================================
# Output models
# ===========================================================================

class RecordingFindingType(str, Enum):
    """The kinds of recording/broadcast issue the module detects (spec bullets)."""

    LOUDNESS_IMBALANCE = "loudness_imbalance"
    VOCAL_CLARITY = "vocal_clarity"
    MIX_INCONSISTENCY = "mix_inconsistency"


class RecordingFinding(BaseModel):
    """A single difference between the recording feed and the room."""

    kind: RecordingFindingType
    severity: Priority
    confidence: float
    message: str


class RecordingAnalysis(BaseModel):
    """The result of comparing the recording feed against the room.

    ``available`` is False when the recording feed could not be analysed this
    cycle (e.g. the recorder input was lost); the UI then shows the feed as
    unavailable rather than silently claiming the broadcast is fine.
    """

    available: bool = True
    findings: List[RecordingFinding] = Field(default_factory=list)
    room_metrics: Optional[AudioMetrics] = None
    recording_metrics: Optional[AudioMetrics] = None

    @property
    def all_clear(self) -> bool:
        """True when the feed was analysed and matches the room well."""
        return self.available and not self.findings


# ===========================================================================
# Strategy interface
# ===========================================================================

class RecordingComparator(ABC):
    """Interface for one recording-vs-room comparison (Strategy Pattern)."""

    @abstractmethod
    def compare(
        self, room: AudioMetrics, recording: AudioMetrics
    ) -> Optional[RecordingFinding]:
        """Return a finding if the feed differs from the room, else None."""


# ===========================================================================
# Comparators (one per spec bullet)
# ===========================================================================

class LoudnessImbalanceComparator(RecordingComparator):
    """Flags the recording feed being louder/quieter than the room."""

    def compare(
        self, room: AudioMetrics, recording: AudioMetrics
    ) -> Optional[RecordingFinding]:
        diff = recording.loudness_lufs - room.loudness_lufs
        magnitude = abs(diff)
        if magnitude < LOUDNESS_IMBALANCE_LUFS:
            return None
        louder = diff > 0
        direction = "louder" if louder else "quieter"
        severity = (
            Priority.HIGH if magnitude >= _LOUDNESS_HIGH_LUFS else Priority.MEDIUM
        )
        # Confidence ramps from 0.5 at the threshold to 1.0 at ~+9 LUFS over it.
        confidence = min(
            1.0, 0.5 + (magnitude - LOUDNESS_IMBALANCE_LUFS) / 12.0
        )
        message = (
            f"The recording/stream feed is about {magnitude:.0f} dB {direction} "
            f"than the room. Adjust the recording bus level to match the house."
        )
        return RecordingFinding(
            kind=RecordingFindingType.LOUDNESS_IMBALANCE,
            severity=severity,
            confidence=round(confidence, 3),
            message=message,
        )


class VocalClarityComparator(RecordingComparator):
    """Flags presence/high-frequency loss on the feed (vocals less clear)."""

    def compare(
        self, room: AudioMetrics, recording: AudioMetrics
    ) -> Optional[RecordingFinding]:
        gap = _energy_gap(room.high_freq_energy, recording.high_freq_energy)
        if gap >= 0:
            # The feed has the same or more presence than the room — not a loss.
            return None
        drop = -gap  # 1 or 2 levels lower
        severity = Priority.HIGH if drop >= 2 else Priority.MEDIUM
        confidence = 0.9 if drop >= 2 else 0.6
        message = (
            "Vocals/presence sound less clear on the recording than in the room. "
            "Add a little high-frequency presence to the recording/stream feed."
        )
        return RecordingFinding(
            kind=RecordingFindingType.VOCAL_CLARITY,
            severity=severity,
            confidence=confidence,
            message=message,
        )


class MixInconsistencyComparator(RecordingComparator):
    """Flags a low-mid tonal-balance mismatch between the feed and the room."""

    def compare(
        self, room: AudioMetrics, recording: AudioMetrics
    ) -> Optional[RecordingFinding]:
        gap = _energy_gap(room.low_mid_energy, recording.low_mid_energy)
        if gap == 0:
            return None
        muddier = gap > 0
        descriptor = "muddier (more low-mid)" if muddier else "thinner (less low-mid)"
        severity = Priority.HIGH if abs(gap) >= 2 else Priority.MEDIUM
        confidence = 0.85 if abs(gap) >= 2 else 0.6
        message = (
            f"The recording mix sounds {descriptor} than the room. "
            "Check the low-mid balance on the recording/stream bus."
        )
        return RecordingFinding(
            kind=RecordingFindingType.MIX_INCONSISTENCY,
            severity=severity,
            confidence=confidence,
            message=message,
        )


# ===========================================================================
# Engine
# ===========================================================================

class RecordingAnalysisEngine:
    """Runs a set of comparators and merges their findings (Strategy host)."""

    def __init__(self, comparators: List[RecordingComparator]) -> None:
        self._comparators = list(comparators)

    @classmethod
    def default(cls) -> "RecordingAnalysisEngine":
        """The standard engine: one comparator per spec bullet."""
        return cls([
            LoudnessImbalanceComparator(),
            VocalClarityComparator(),
            MixInconsistencyComparator(),
        ])

    def analyze(
        self, room: AudioMetrics, recording: AudioMetrics
    ) -> RecordingAnalysis:
        """Compare the recording feed against the room (failsafe, sorted)."""
        findings: List[RecordingFinding] = []
        for comparator in self._comparators:
            try:
                finding = comparator.compare(room, recording)
            except Exception:  # noqa: BLE001 — failsafe: skip a broken comparator
                continue
            if finding is not None:
                findings.append(finding)

        findings.sort(
            key=lambda f: (_PRIORITY_RANK.get(f.severity, 0), f.confidence),
            reverse=True,
        )
        return RecordingAnalysis(
            available=True,
            findings=findings,
            room_metrics=room,
            recording_metrics=recording,
        )
