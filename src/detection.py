"""
src/detection.py

Agent 3: Detection Engine for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt — AGENT 3):
  - "Combine audio + mixer data to detect sound issues"
  - Detect: Feedback risk, Vocal masking, Clipping, Loudness issues,
            EQ problems (mud, harshness)
  - "Assign priority" / "Calculate confidence score"
  - Output Example:
        {"issue": "vocal_masking", "channel": "Lead Vocal",
         "priority": "high", "confidence": 0.9}
  - Confidence: High (>0.8) / Medium (0.5-0.8) / Low (<0.5)
  - Priority order: 1 Feedback (critical), 2 Vocal clarity
        (Sermon highest, Worship next), 3 Clipping, 4 Loudness, 5 Balance

Design — STRATEGY PATTERN
-------------------------
Each issue type is found by an interchangeable ``IssueDetector``. The
``DetectionEngine`` depends only on that abstraction: it runs every detector
over a ``DetectionContext`` (Agent 1 audio + Agent 2 mixer + Profile), then
returns the detected issues sorted by spec priority and confidence. New issue
types can be added by writing a detector — the engine never changes.

Failsafe (spec: "System must never crash during service"): a detector that
raises is skipped, never propagated.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.audio_analysis import AudioMetrics, EnergyLevel
from src.mixer_state import ChannelState, MixerState
from src.state_manager import LoudnessMode, Scene


# ===========================================================================
# Enumerations
# ===========================================================================

class IssueType(str, Enum):
    """Categories of sound issue the engine can detect (spec issue list)."""

    FEEDBACK = "feedback"
    VOCAL_MASKING = "vocal_masking"
    CLIPPING = "clipping"
    LOUDNESS = "loudness"
    EQ_MUD = "eq_mud"
    EQ_HARSHNESS = "eq_harshness"
    BALANCE = "balance"


class Priority(str, Enum):
    """Priority tiers for a detected issue."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceLevel(str, Enum):
    """Categorical confidence band (spec: High / Medium / Low)."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ===========================================================================
# Tuning constants
# ===========================================================================

#: Peak (dBFS) at/above which clipping is flagged.
CLIP_THRESHOLD_DBFS = -1.0
#: Window (dB) over which clipping confidence ramps 0 -> 1.
_CLIP_CONF_RANGE = 3.0

#: Integrated-loudness targets (LUFS) per loudness mode.
LOUDNESS_TARGETS: Dict[LoudnessMode, float] = {
    LoudnessMode.CONSERVATIVE: -23.0,
    LoudnessMode.BALANCED: -18.0,
    LoudnessMode.ENERGETIC: -14.0,
}
#: Allowed deviation (dB) around the loudness target before flagging.
LOUDNESS_TOLERANCE_DB = 3.0
#: Deviation (dB) beyond tolerance that maps to full confidence.
_LOUDNESS_CONF_RANGE = 6.0

#: Peak (dBFS) above which sustained high-frequency energy reads as feedback.
FEEDBACK_PEAK_DBFS = -3.0

#: Fader (dB) at/below which a vocal is considered "not prominent".
VOCAL_PROMINENT_DB = -6.0
#: Fader depth (dB) below the prominence line that maps to full confidence.
_VOCAL_CONF_RANGE = 12.0

#: Confidence assigned to categorical EQ findings.
_EQ_CONFIDENCE = 0.7


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ===========================================================================
# Confidence + priority helpers
# ===========================================================================

def confidence_level(score: float) -> ConfidenceLevel:
    """Map a 0..1 confidence score to a categorical band (spec thresholds)."""
    if score > 0.8:
        return ConfidenceLevel.HIGH
    if score >= 0.5:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def priority_for(issue_type: IssueType, scene: Scene) -> Priority:
    """Return the display priority tier for an issue type (scene-aware)."""
    if issue_type == IssueType.FEEDBACK:
        return Priority.CRITICAL
    if issue_type == IssueType.VOCAL_MASKING:
        return Priority.HIGH
    if issue_type == IssueType.CLIPPING:
        return Priority.HIGH
    if issue_type in (IssueType.LOUDNESS, IssueType.EQ_MUD, IssueType.EQ_HARSHNESS):
        return Priority.MEDIUM
    return Priority.LOW


#: Base ordering ranks that encode the spec priority list (higher = sooner).
_BASE_RANK: Dict[IssueType, int] = {
    IssueType.FEEDBACK: 100,
    IssueType.VOCAL_MASKING: 90,  # adjusted by scene below
    IssueType.CLIPPING: 80,
    IssueType.LOUDNESS: 70,
    IssueType.EQ_HARSHNESS: 62,
    IssueType.EQ_MUD: 60,
    IssueType.BALANCE: 50,
}

#: Scene weighting for vocal clarity (spec: Sermon highest, Worship next).
_VOCAL_SCENE_BONUS: Dict[Scene, int] = {
    Scene.SERMON: 8,
    Scene.WORSHIP: 4,
    Scene.CUSTOM: 2,
    Scene.PLAY: 0,
}


def spec_rank(issue_type: IssueType, scene: Scene) -> int:
    """Sorting rank for an issue under a given scene (higher sorts first)."""
    rank = _BASE_RANK[issue_type]
    if issue_type == IssueType.VOCAL_MASKING:
        rank += _VOCAL_SCENE_BONUS.get(scene, 0)
    return rank


# ===========================================================================
# Models
# ===========================================================================

class Issue(BaseModel):
    """A single detected sound issue (spec output shape)."""

    issue: IssueType
    channel: Optional[str]
    priority: Priority
    confidence: float

    @property
    def confidence_label(self) -> ConfidenceLevel:
        """The categorical High/Medium/Low band for this issue's confidence."""
        return confidence_level(self.confidence)


class Profile(BaseModel):
    """Environment + user-preference data applied during detection.

    Spec (Agent 5 / overrides): scene + loudness mode steer thresholds and
    priority; suppressed_issues lets the user silence categories they have
    marked as normal (e.g. "ignore choir balance").
    """

    scene: Scene = Scene.SERMON
    loudness_mode: LoudnessMode = LoudnessMode.CONSERVATIVE
    suppressed_issues: List[IssueType] = Field(default_factory=list)
    vocal_keywords: List[str] = Field(
        default_factory=lambda: ["vocal", "vox", "lead", "speech", "pastor", "preach"]
    )


@dataclass(frozen=True)
class DetectionContext:
    """Combined inputs handed to every detector for a single analysis pass."""

    audio: AudioMetrics
    mixer: MixerState
    profile: Profile


# ===========================================================================
# Strategy interface
# ===========================================================================

class IssueDetector(ABC):
    """Interface for a single, interchangeable issue-detection algorithm."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used for logging / introspection."""

    @abstractmethod
    def detect(self, context: DetectionContext) -> Optional[Issue]:
        """Return an Issue if this detector fires, else None."""


# ===========================================================================
# Concrete detectors
# ===========================================================================

class ClippingDetector(IssueDetector):
    """Flags peaks at/near digital full scale (spec: Clipping)."""

    @property
    def name(self) -> str:
        return "clipping"

    def detect(self, context: DetectionContext) -> Optional[Issue]:
        peak = context.audio.peak_level
        if peak < CLIP_THRESHOLD_DBFS:
            return None
        confidence = _clamp(
            (peak - (CLIP_THRESHOLD_DBFS - _CLIP_CONF_RANGE)) / _CLIP_CONF_RANGE,
            0.0,
            1.0,
        )
        return Issue(
            issue=IssueType.CLIPPING,
            channel=None,
            priority=priority_for(IssueType.CLIPPING, context.profile.scene),
            confidence=confidence,
        )


class LoudnessDetector(IssueDetector):
    """Flags integrated loudness that drifts off the mode target (spec: Loudness)."""

    @property
    def name(self) -> str:
        return "loudness"

    def detect(self, context: DetectionContext) -> Optional[Issue]:
        target = LOUDNESS_TARGETS[context.profile.loudness_mode]
        deviation = context.audio.loudness_lufs - target
        beyond = abs(deviation) - LOUDNESS_TOLERANCE_DB
        if beyond <= 0.0:
            return None
        confidence = _clamp(beyond / _LOUDNESS_CONF_RANGE, 0.0, 1.0)
        return Issue(
            issue=IssueType.LOUDNESS,
            channel=None,
            priority=priority_for(IssueType.LOUDNESS, context.profile.scene),
            confidence=confidence,
        )


class EqMudDetector(IssueDetector):
    """Flags excessive low-mid energy (spec: EQ problems — mud)."""

    @property
    def name(self) -> str:
        return "eq_mud"

    def detect(self, context: DetectionContext) -> Optional[Issue]:
        if context.audio.low_mid_energy != EnergyLevel.HIGH:
            return None
        return Issue(
            issue=IssueType.EQ_MUD,
            channel=None,
            priority=priority_for(IssueType.EQ_MUD, context.profile.scene),
            confidence=_EQ_CONFIDENCE,
        )


class EqHarshnessDetector(IssueDetector):
    """Flags excessive high-frequency energy (spec: EQ problems — harshness)."""

    @property
    def name(self) -> str:
        return "eq_harshness"

    def detect(self, context: DetectionContext) -> Optional[Issue]:
        if context.audio.high_freq_energy != EnergyLevel.HIGH:
            return None
        return Issue(
            issue=IssueType.EQ_HARSHNESS,
            channel=None,
            priority=priority_for(IssueType.EQ_HARSHNESS, context.profile.scene),
            confidence=_EQ_CONFIDENCE,
        )


class FeedbackDetector(IssueDetector):
    """Flags sustained, loud high-frequency energy as feedback risk (spec: rank 1).

    With the coarse metrics available in V1, feedback is approximated as high
    high-frequency energy combined with a hot peak. This is the critical-tier
    issue, so it is intentionally sensitive.
    """

    @property
    def name(self) -> str:
        return "feedback"

    def detect(self, context: DetectionContext) -> Optional[Issue]:
        audio = context.audio
        if audio.high_freq_energy != EnergyLevel.HIGH:
            return None
        if audio.peak_level < FEEDBACK_PEAK_DBFS:
            return None
        depth = _clamp((audio.peak_level - FEEDBACK_PEAK_DBFS) / 3.0, 0.0, 1.0)
        confidence = _clamp(0.6 + 0.4 * depth, 0.0, 1.0)
        return Issue(
            issue=IssueType.FEEDBACK,
            channel=None,
            priority=priority_for(IssueType.FEEDBACK, context.profile.scene),
            confidence=confidence,
        )


class VocalMaskingDetector(IssueDetector):
    """Flags vocals buried under music — combines mixer + audio (spec flagship).

    Fires when the audio shows heavy low-mid (music) energy *and* the mixer's
    vocal channel is unmuted but pulled down. The two corroborating sources
    (board fader + room audio) are the spec's "agreement between sources"
    confidence factor: the lower the vocal sits, the higher the confidence.
    """

    @property
    def name(self) -> str:
        return "vocal_masking"

    def detect(self, context: DetectionContext) -> Optional[Issue]:
        if context.audio.low_mid_energy != EnergyLevel.HIGH:
            return None

        vocal = self._find_vocal_channel(context.mixer, context.profile)
        if vocal is None or vocal.fader > VOCAL_PROMINENT_DB:
            return None

        depth = _clamp(
            (VOCAL_PROMINENT_DB - vocal.fader) / _VOCAL_CONF_RANGE, 0.0, 1.0
        )
        confidence = _clamp(0.6 + 0.4 * depth, 0.0, 1.0)
        return Issue(
            issue=IssueType.VOCAL_MASKING,
            channel=vocal.name,
            priority=priority_for(IssueType.VOCAL_MASKING, context.profile.scene),
            confidence=confidence,
        )

    @staticmethod
    def _find_vocal_channel(
        mixer: MixerState, profile: Profile
    ) -> Optional[ChannelState]:
        for channel in mixer.channels.values():
            if channel.mute:
                continue
            lowered = channel.name.lower()
            if any(keyword in lowered for keyword in profile.vocal_keywords):
                return channel
        return None


# ===========================================================================
# Engine
# ===========================================================================

class DetectionEngine:
    """Runs issue detectors and returns prioritised, profile-filtered issues."""

    def __init__(self, detectors: List[IssueDetector]) -> None:
        self._detectors: List[IssueDetector] = list(detectors)

    @classmethod
    def default(cls) -> "DetectionEngine":
        """Build the standard Agent 3 detector set (ordered by spec priority)."""
        return cls(
            [
                FeedbackDetector(),
                VocalMaskingDetector(),
                ClippingDetector(),
                LoudnessDetector(),
                EqHarshnessDetector(),
                EqMudDetector(),
            ]
        )

    @property
    def detectors(self) -> List[IssueDetector]:
        return list(self._detectors)

    def detect(
        self,
        audio: AudioMetrics,
        mixer: MixerState,
        profile: Optional[Profile] = None,
    ) -> List[Issue]:
        """Combine inputs, run every detector, and return sorted issues.

        Issues whose type is suppressed by the profile are dropped. A detector
        that raises is skipped (failsafe), so one fault never aborts the pass.
        """
        profile = profile or Profile()
        context = DetectionContext(audio=audio, mixer=mixer, profile=profile)

        found: List[Issue] = []
        for detector in self._detectors:
            try:
                issue = detector.detect(context)
            except Exception:  # noqa: BLE001 — failsafe: never crash during service
                continue
            if issue is None:
                continue
            if issue.issue in profile.suppressed_issues:
                continue
            found.append(issue)

        found.sort(
            key=lambda i: (spec_rank(i.issue, profile.scene), i.confidence),
            reverse=True,
        )
        return found
