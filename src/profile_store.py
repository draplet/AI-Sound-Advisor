"""
src/profile_store.py

Agent 5: Context & Profile Agent for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt — AGENT 5):
  - "Manage profiles (load/save)"  → JSON persistence (V1 storage spec)
  - "Store user instructions and overrides"
  - "Track 'normal' conditions"  → mark_normal()
  - "Apply learned behavior"  → detection_profile()
  - Profile Contents: event type, room size/characteristics, acoustic sources,
                      user overrides
  - Profile Management: real-time learning + end-of-event "summary of learned
                        changes" (pending_changes) before saving
  - Calibration Mode: capture a "good mix" baseline from Agent 1 metrics

Design
------
``EnvironmentProfile`` is the persisted master profile. ``ContextProfileAgent``
owns the active profile, persists it as JSON, and records every learned change
so the UI can show a summary at the end of an event before saving. The agent
bridges to Agent 3 by emitting a ``detection.Profile`` via ``detection_profile``
— so user overrides (e.g. marking an issue as normal) actually steer detection.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.audio_analysis import AudioMetrics, EnergyLevel
from src.detection import IssueType, Profile as DetectionProfile
from src.state_manager import LoudnessMode, Scene

#: Default channel-name keywords used to identify vocal channels (mirrors the
#: detection default so a fresh profile behaves like the engine default).
_DEFAULT_VOCAL_KEYWORDS = ["vocal", "vox", "lead", "speech", "pastor", "preach"]


class RoomSize(str, Enum):
    """Coarse room-size descriptor for the environment profile."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# ===========================================================================
# Models
# ===========================================================================

class CalibrationBaseline(BaseModel):
    """A captured "good mix" reference for this environment (calibration mode)."""

    loudness_lufs: float
    low_mid_energy: EnergyLevel
    high_freq_energy: EnergyLevel

    @classmethod
    def from_audio(cls, metrics: AudioMetrics) -> "CalibrationBaseline":
        """Build a baseline from an Agent 1 AudioMetrics snapshot."""
        return cls(
            loudness_lufs=metrics.loudness_lufs,
            low_mid_energy=metrics.low_mid_energy,
            high_freq_energy=metrics.high_freq_energy,
        )


class EnvironmentProfile(BaseModel):
    """Persisted profile: environment configuration + learned user knowledge."""

    name: str
    event_type: Scene = Scene.SERMON
    loudness_mode: LoudnessMode = LoudnessMode.CONSERVATIVE
    room_size: RoomSize = RoomSize.MEDIUM
    room_characteristics: str = ""
    acoustic_sources: List[str] = Field(default_factory=list)
    user_instructions: List[str] = Field(default_factory=list)
    overrides: Dict[str, str] = Field(default_factory=dict)
    suppressed_issues: List[IssueType] = Field(default_factory=list)
    vocal_keywords: List[str] = Field(
        default_factory=lambda: list(_DEFAULT_VOCAL_KEYWORDS)
    )
    calibration: Optional[CalibrationBaseline] = None

    def detection_profile(self) -> DetectionProfile:
        """Project this profile into the Profile that Agent 3 consumes."""
        return DetectionProfile(
            scene=self.event_type,
            loudness_mode=self.loudness_mode,
            suppressed_issues=list(self.suppressed_issues),
            vocal_keywords=list(self.vocal_keywords),
        )


# ===========================================================================
# Agent
# ===========================================================================

class ContextProfileAgent:
    """Owns the active profile, persists it, and tracks learned changes.

    ``storage_dir`` is where profiles are read/written as ``<name>.json``.
    Use :meth:`in_memory` for a transient agent (no disk) in tests/calibration.
    """

    def __init__(
        self,
        storage_dir: Optional[Path | str],
        profile: Optional[EnvironmentProfile] = None,
    ) -> None:
        self._storage_dir = Path(storage_dir) if storage_dir is not None else None
        self._active = profile or EnvironmentProfile(name="default")
        self._changes: List[str] = []

    @classmethod
    def in_memory(cls, name: str = "default") -> "ContextProfileAgent":
        """Create an agent with no backing directory (save() is unavailable)."""
        return cls(storage_dir=None, profile=EnvironmentProfile(name=name))

    # ------------------------------------------------------------------
    # Active profile + change tracking
    # ------------------------------------------------------------------

    @property
    def active(self) -> EnvironmentProfile:
        return self._active

    @property
    def has_unsaved_changes(self) -> bool:
        return bool(self._changes)

    def pending_changes(self) -> List[str]:
        """Human-readable summary of changes learned since load/save."""
        return list(self._changes)

    def _record(self, message: str) -> None:
        self._changes.append(message)

    # ------------------------------------------------------------------
    # Persistence (JSON — V1 storage)
    # ------------------------------------------------------------------

    def _path_for(self, name: str) -> Path:
        if self._storage_dir is None:
            raise RuntimeError("this agent has no storage directory configured")
        return self._storage_dir / f"{name}.json"

    def save(self) -> Path:
        """Persist the active profile to JSON and clear pending changes."""
        path = self._path_for(self._active.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._active.model_dump_json(indent=2), encoding="utf-8")
        self._changes.clear()
        return path

    def load(self, name: str) -> EnvironmentProfile:
        """Load a profile by name, making it active. Raises if it is missing."""
        path = self._path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"no profile named {name!r} in {self._storage_dir}")
        self._active = EnvironmentProfile.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        self._changes.clear()
        return self._active

    def load_or_create(self, name: str) -> EnvironmentProfile:
        """Load a profile if it exists, otherwise start a fresh one by that name."""
        try:
            return self.load(name)
        except FileNotFoundError:
            self._active = EnvironmentProfile(name=name)
            self._changes.clear()
            return self._active

    def list_profiles(self) -> List[str]:
        """Names of all saved profiles in the storage directory."""
        if self._storage_dir is None or not self._storage_dir.exists():
            return []
        return sorted(p.stem for p in self._storage_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Real-time learning (mutators record a change for the summary)
    # ------------------------------------------------------------------

    def set_event_type(self, scene: Scene) -> None:
        self._active.event_type = scene
        self._record(f"Event type set to {scene.value}.")

    def set_loudness_mode(self, mode: LoudnessMode) -> None:
        self._active.loudness_mode = mode
        self._record(f"Loudness mode set to {mode.value}.")

    def set_room(self, size: RoomSize, characteristics: str = "") -> None:
        self._active.room_size = size
        if characteristics:
            self._active.room_characteristics = characteristics
        self._record(f"Room set to {size.value}.")

    def add_acoustic_source(self, source: str) -> None:
        if source not in self._active.acoustic_sources:
            self._active.acoustic_sources.append(source)
            self._record(f"Added acoustic source: {source}.")

    def add_user_instruction(self, text: str) -> None:
        self._active.user_instructions.append(text)
        self._record(f"Noted instruction: '{text}'.")

    def set_override(self, key: str, value: str) -> None:
        self._active.overrides[key] = value
        self._record(f"Override: {key} = {value}.")

    def mark_normal(self, issue_type: IssueType) -> None:
        """Mark an issue category as a normal condition (suppress it)."""
        if issue_type not in self._active.suppressed_issues:
            self._active.suppressed_issues.append(issue_type)
            self._record(f"Marked {issue_type.value} as normal (will be suppressed).")

    # ------------------------------------------------------------------
    # Calibration mode
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._active.calibration is not None

    def calibrate(self, metrics: AudioMetrics) -> CalibrationBaseline:
        """Capture a "good mix" baseline from current audio metrics."""
        baseline = CalibrationBaseline.from_audio(metrics)
        self._active.calibration = baseline
        self._record(
            f"Captured calibration baseline at {baseline.loudness_lufs:.1f} LUFS."
        )
        return baseline

    # ------------------------------------------------------------------
    # Apply learned behavior
    # ------------------------------------------------------------------

    def detection_profile(self) -> DetectionProfile:
        """Emit the detection Profile reflecting all learned behavior."""
        return self._active.detection_profile()
