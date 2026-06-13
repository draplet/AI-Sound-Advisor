"""
src/state_manager.py

Thread-safe system state registry for the AI Sound Advisor.

Spec requirements implemented here:
  - "thread-safe registry of: current_scene, loudness_state,
     active_issues, ignored_issues"
  - "System detects or allows selection: Sermon / Worship / Play / Custom"
  - "Loudness Modes: Conservative / Balanced / Energetic"
  - Ignored issues are suppressed and must not re-enter the active set.
"""
from __future__ import annotations

import threading
from enum import Enum
from typing import Dict


class Scene(str, Enum):
    """Selectable mixing scenes (spec default: SERMON)."""

    SERMON = "sermon"
    WORSHIP = "worship"
    PLAY = "play"
    CUSTOM = "custom"


class LoudnessMode(str, Enum):
    """Loudness profiles (spec default: CONSERVATIVE)."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    ENERGETIC = "energetic"


class SystemStateManager:
    """
    Central, thread-safe registry of the live system state.

    All mutations are guarded by a reentrant lock so concurrent writers
    (e.g. multiple analysis threads) cannot corrupt the issue registries.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.current_scene: Scene = Scene.SERMON
        self.loudness_mode: LoudnessMode = LoudnessMode.CONSERVATIVE
        self.active_issues: Dict[str, dict] = {}
        self.ignored_issues: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Issue registry
    # ------------------------------------------------------------------

    def add_issue(self, issue_id: str, issue_type: str, channel: str) -> None:
        """
        Register an issue in active_issues.

        An issue that has already been ignored stays suppressed: re-adding the
        same issue_id is a no-op so dismissed warnings do not reappear.
        """
        with self._lock:
            if issue_id in self.ignored_issues:
                return
            self.active_issues[issue_id] = {
                "issue_type": issue_type,
                "channel": channel,
            }

    def ignore_issue(self, issue_id: str) -> None:
        """Suppress an issue: remove it from active and record it as ignored."""
        with self._lock:
            entry = self.active_issues.pop(issue_id, {"issue_type": None, "channel": None})
            self.ignored_issues[issue_id] = entry

    def resolve_issue(self, issue_id: str) -> None:
        """Remove an issue from the active registry once it is resolved."""
        with self._lock:
            self.active_issues.pop(issue_id, None)

    def get_active_issues(self) -> Dict[str, dict]:
        """
        Return a snapshot of active, non-ignored issues.

        A copy is returned so callers can iterate safely while other threads
        continue to mutate the live registry.
        """
        with self._lock:
            return {
                issue_id: entry
                for issue_id, entry in self.active_issues.items()
                if issue_id not in self.ignored_issues
            }

    # ------------------------------------------------------------------
    # Scene / loudness selection
    # ------------------------------------------------------------------

    def set_scene(self, scene: Scene) -> None:
        """Update the current mixing scene."""
        with self._lock:
            self.current_scene = scene

    def set_loudness_mode(self, mode: LoudnessMode) -> None:
        """Update the active loudness profile."""
        with self._lock:
            self.loudness_mode = mode
