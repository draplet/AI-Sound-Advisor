"""
src/audio_analysis.py

Agent 1: Audio Signal Analysis Engine for the AI Sound Advisor.

Spec requirements implemented here (project_master_plan.txt — AGENT 1):
  - "Calculate LUFS loudness using pyloudnorm"  → LoudnessStrategy
  - "Perform FFT frequency analysis using librosa"  → SpectralBandStrategy
  - "Detect peaks using numpy/scipy"  → PeakStrategy
  - "Output structured metrics"  → AudioMetrics (Pydantic)

Output shape (spec example):
    {
      "loudness_lufs": -15,
      "peak_level": -1.2,
      "low_mid_energy": "high",
      "high_freq_energy": "normal"
    }

Design — STRATEGY PATTERN
-------------------------
Each metric is produced by an interchangeable ``AnalysisStrategy``. The
``AudioAnalysisEngine`` depends only on that abstraction: it runs whatever
strategies it is given and merges their dict fragments into one validated
``AudioMetrics`` object. New analyses (e.g. crest factor, spectral centroid)
can be added — or the loudness algorithm swapped for a different one — without
touching the engine.

Dependency note
---------------
The spec names pyloudnorm / librosa / scipy, but only numpy is guaranteed in
this environment. The DSP is therefore implemented directly in numpy so the
engine is self-contained and deterministic:
  - Loudness is a genuine ITU-R BS.1770-4 *integrated* loudness measurement
    (K-weighting pre-filter + 400 ms / 75 %-overlap blocks + absolute and
    relative gating) — the same algorithm pyloudnorm implements.
  - FFT band analysis uses numpy.fft.
  - Peak detection uses numpy.

Performance: analysis is intended for short real-time buffers (≤ a few seconds),
satisfying the spec's "real time (≤ 50 ms per cycle)" target for live use.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Reported peak/loudness when a buffer is effectively silent. Real digital
#: silence is -inf dB; we clamp to a finite floor so the metrics serialise.
PEAK_FLOOR_DBFS: float = -120.0
LOUDNESS_FLOOR_LUFS: float = -120.0

#: Default frequency bands (Hz) for spectral energy classification.
DEFAULT_LOW_MID_BAND: Tuple[float, float] = (100.0, 1_000.0)
DEFAULT_HIGH_BAND: Tuple[float, float] = (4_000.0, 16_000.0)

#: Band-energy fraction thresholds (fraction of total spectral energy).
DEFAULT_HIGH_FRACTION: float = 0.40   # >= this  → HIGH
DEFAULT_LOW_FRACTION: float = 0.10    # <  this  → LOW  (else NORMAL)

#: BS.1770 integrated-loudness gating parameters.
_BLOCK_S = 0.400          # 400 ms measurement block
_BLOCK_OVERLAP = 0.75     # 75 % overlap → 100 ms hop
_ABSOLUTE_GATE_LUFS = -70.0
_RELATIVE_GATE_DB = -10.0
_LUFS_OFFSET = -0.691     # BS.1770 calibration constant


# ===========================================================================
# Structured output (spec: "Output structured metrics")
# ===========================================================================

class EnergyLevel(str, Enum):
    """Categorical energy level for a frequency band."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class AudioMetrics(BaseModel):
    """Validated structured metrics emitted by Agent 1 (spec output shape)."""

    loudness_lufs: float
    peak_level: float
    low_mid_energy: EnergyLevel
    high_freq_energy: EnergyLevel


# ===========================================================================
# Strategy Pattern interface
# ===========================================================================

class AnalysisStrategy(ABC):
    """Interface for a single, interchangeable audio-analysis algorithm.

    A strategy receives a mono float64 buffer plus its sample rate and returns
    a dict fragment whose keys are a subset of ``AudioMetrics`` fields.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used for logging / introspection."""

    @abstractmethod
    def analyze(self, samples: np.ndarray, sample_rate: int) -> Dict[str, object]:
        """Compute this strategy's metric fragment for ``samples``."""


# ===========================================================================
# Concrete strategies
# ===========================================================================

class PeakStrategy(AnalysisStrategy):
    """Peak level in dBFS (0 dBFS == digital full scale). Spec: "Detect peaks"."""

    @property
    def name(self) -> str:
        return "peak"

    def analyze(self, samples: np.ndarray, sample_rate: int) -> Dict[str, object]:
        samples = np.asarray(samples, dtype=np.float64)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak <= 0.0:
            return {"peak_level": PEAK_FLOOR_DBFS}
        return {"peak_level": 20.0 * math.log10(peak)}


class LoudnessStrategy(AnalysisStrategy):
    """ITU-R BS.1770-4 *integrated* loudness in LUFS. Spec: "Calculate LUFS"."""

    @property
    def name(self) -> str:
        return "loudness"

    def analyze(self, samples: np.ndarray, sample_rate: int) -> Dict[str, object]:
        samples = np.asarray(samples, dtype=np.float64)
        return {"loudness_lufs": _integrated_loudness(samples, sample_rate)}


class SpectralBandStrategy(AnalysisStrategy):
    """FFT band-energy classification. Spec: "FFT frequency analysis".

    Reports each band's share of total spectral energy as LOW / NORMAL / HIGH.
    Bands and thresholds are injectable for tuning and testing.
    """

    def __init__(
        self,
        low_mid_band: Tuple[float, float] = DEFAULT_LOW_MID_BAND,
        high_band: Tuple[float, float] = DEFAULT_HIGH_BAND,
        high_fraction: float = DEFAULT_HIGH_FRACTION,
        low_fraction: float = DEFAULT_LOW_FRACTION,
    ) -> None:
        self.low_mid_band = low_mid_band
        self.high_band = high_band
        self.high_fraction = high_fraction
        self.low_fraction = low_fraction

    @property
    def name(self) -> str:
        return "spectral_band"

    def analyze(self, samples: np.ndarray, sample_rate: int) -> Dict[str, object]:
        samples = np.asarray(samples, dtype=np.float64)

        # Power spectrum via real FFT; exclude the DC bin from the total.
        spectrum = np.abs(np.fft.rfft(samples)) ** 2
        freqs = np.fft.rfftfreq(samples.shape[0], d=1.0 / sample_rate)
        total = float(np.sum(spectrum[1:]))

        low_mid = self._classify(self._band_fraction(spectrum, freqs, total,
                                                      self.low_mid_band))
        high = self._classify(self._band_fraction(spectrum, freqs, total,
                                                   self.high_band))
        return {"low_mid_energy": low_mid, "high_freq_energy": high}

    @staticmethod
    def _band_fraction(
        spectrum: np.ndarray,
        freqs: np.ndarray,
        total: float,
        band: Tuple[float, float],
    ) -> float:
        if total <= 0.0:
            return 0.0
        lo, hi = band
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(spectrum[mask]) / total)

    def _classify(self, fraction: float) -> EnergyLevel:
        if fraction >= self.high_fraction:
            return EnergyLevel.HIGH
        if fraction < self.low_fraction:
            return EnergyLevel.LOW
        return EnergyLevel.NORMAL


# ===========================================================================
# Engine
# ===========================================================================

class AudioAnalysisEngine:
    """Runs a set of ``AnalysisStrategy`` objects and aggregates the result.

    The engine validates and normalises the input buffer once (mono float64),
    then lets each strategy contribute fields to a single ``AudioMetrics``.
    Strategies run in order; a later strategy may override an earlier field.
    """

    def __init__(self, strategies: Sequence[AnalysisStrategy]) -> None:
        self._strategies: List[AnalysisStrategy] = list(strategies)

    @classmethod
    def default(cls) -> "AudioAnalysisEngine":
        """Build the standard Agent 1 engine (loudness + peak + spectral)."""
        return cls([LoudnessStrategy(), PeakStrategy(), SpectralBandStrategy()])

    @property
    def strategies(self) -> Tuple[AnalysisStrategy, ...]:
        """Immutable view of the configured strategies."""
        return tuple(self._strategies)

    def analyze(self, samples: np.ndarray, sample_rate: int) -> AudioMetrics:
        """Analyse ``samples`` and return validated structured metrics."""
        mono = _prepare_samples(samples)
        merged: Dict[str, object] = {}
        for strategy in self._strategies:
            merged.update(strategy.analyze(mono, sample_rate))
        return AudioMetrics(**merged)


# ===========================================================================
# Input preparation
# ===========================================================================

def _prepare_samples(samples: np.ndarray) -> np.ndarray:
    """Validate and normalise an input buffer to a 1-D float64 mono array.

    - 2-D input is treated as (frames, channels) and downmixed by averaging.
    - Empty, >2-D, or non-finite (NaN/Inf) buffers raise ValueError so a bad
      buffer surfaces as a clear precondition error to the caller.
    """
    arr = np.asarray(samples, dtype=np.float64)

    if arr.size == 0:
        raise ValueError("audio buffer is empty")
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    elif arr.ndim > 2:
        raise ValueError(
            "audio buffer must be 1-D (mono) or 2-D (frames, channels), "
            f"got {arr.ndim}-D"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("audio buffer contains non-finite samples (NaN/Inf)")
    return arr


# ===========================================================================
# BS.1770 loudness internals (pure numpy)
# ===========================================================================

def _k_weighting_coeffs(
    fs: int,
) -> Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]],
           Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Return the two BS.1770 K-weighting biquad stages for sample rate ``fs``.

    Stage 1 is a high-shelf ("head") filter; stage 2 is a high-pass. Coefficients
    are derived analytically per sample rate (matching pyloudnorm). Each stage is
    returned as ((b0, b1, b2), (a0, a1, a2)) with a0 normalised to 1.0.
    """
    # --- Stage 1: high-shelf -------------------------------------------------
    f0 = 1681.974450955533
    gain_db = 3.999843853973347
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / fs)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    den = 1.0 + k / q + k * k
    s1_b = (
        (vh + vb * k / q + k * k) / den,
        2.0 * (k * k - vh) / den,
        (vh - vb * k / q + k * k) / den,
    )
    s1_a = (1.0, 2.0 * (k * k - 1.0) / den, (1.0 - k / q + k * k) / den)

    # --- Stage 2: high-pass --------------------------------------------------
    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = math.tan(math.pi * f0 / fs)
    den = 1.0 + k / q + k * k
    s2_b = (1.0, -2.0, 1.0)
    s2_a = (1.0, 2.0 * (k * k - 1.0) / den, (1.0 - k / q + k * k) / den)

    return (s1_b, s1_a), (s2_b, s2_a)


def _biquad(x: np.ndarray, b: Tuple[float, float, float],
            a: Tuple[float, float, float]) -> np.ndarray:
    """Apply a single biquad (Direct Form I) IIR filter; a[0] assumed 1.0.

    Equivalent to scipy.signal.lfilter(b, a, x) for a 3-tap biquad, implemented
    directly so the module has no scipy dependency.
    """
    b0, b1, b2 = b
    _, a1, a2 = a
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for n in range(x.shape[0]):
        xn = x[n]
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn
    return y


def _integrated_loudness(samples: np.ndarray, fs: int) -> float:
    """Compute BS.1770-4 integrated loudness (LUFS) of a mono buffer.

    Returns ``LOUDNESS_FLOOR_LUFS`` for silence or buffers that gate to empty.
    """
    if samples.size == 0 or fs <= 0:
        return LOUDNESS_FLOOR_LUFS

    # K-weight: high-shelf then high-pass.
    (b1, a1), (b2, a2) = _k_weighting_coeffs(fs)
    weighted = _biquad(_biquad(samples, b1, a1), b2, a2)

    block = int(round(_BLOCK_S * fs))
    hop = max(1, int(round(block * (1.0 - _BLOCK_OVERLAP))))
    n = weighted.shape[0]

    if block <= 0:
        return LOUDNESS_FLOOR_LUFS
    if n < block:
        # Buffer shorter than one gating block: measure it as a single block.
        starts = [0]
        block = n
    else:
        starts = list(range(0, n - block + 1, hop))

    # Mean-square ("z") per block.
    z = np.array([float(np.mean(weighted[s:s + block] ** 2)) for s in starts])

    with np.errstate(divide="ignore"):
        block_lufs = _LUFS_OFFSET + 10.0 * np.log10(z)

    # Absolute gate at -70 LUFS.
    abs_kept = z[block_lufs > _ABSOLUTE_GATE_LUFS]
    if abs_kept.size == 0:
        return LOUDNESS_FLOOR_LUFS

    # Relative gate at (gated mean loudness) - 10 LU.
    relative_gate = _LUFS_OFFSET + 10.0 * math.log10(float(np.mean(abs_kept))) \
        + _RELATIVE_GATE_DB
    rel_kept = abs_kept[
        (_LUFS_OFFSET + 10.0 * np.log10(abs_kept)) > relative_gate
    ]
    if rel_kept.size == 0:
        return LOUDNESS_FLOOR_LUFS

    integrated = _LUFS_OFFSET + 10.0 * math.log10(float(np.mean(rel_kept)))
    return integrated if math.isfinite(integrated) else LOUDNESS_FLOOR_LUFS
