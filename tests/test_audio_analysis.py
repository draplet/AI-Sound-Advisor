"""
tests/test_audio_analysis.py

AI Sound Advisor — Agent 1: Audio Signal Analysis Engine
TDD Test Suite (written BEFORE application code exists)

Specification Reference (project_master_plan.txt — AGENT 1):
  - "Calculate LUFS loudness using pyloudnorm"
  - "Perform FFT frequency analysis using librosa"
  - "Detect peaks using numpy/scipy"
  - "Output structured metrics"
  - Output Example:
        {
          "loudness_lufs": -15,
          "peak_level": -1.2,
          "low_mid_energy": "high",
          "high_freq_energy": "normal"
        }

Design note:
  Agent 1 is built with the STRATEGY PATTERN. Each metric is produced by an
  interchangeable AnalysisStrategy; the AudioAnalysisEngine depends only on
  that abstraction and aggregates strategy outputs into a validated
  AudioMetrics Pydantic model. This suite verifies both the individual
  strategies and the pluggability of the engine.

Modules under test (DO NOT EXIST YET — that is the intended TDD red state):
  src.audio_analysis  →  AudioAnalysisEngine, AudioMetrics, EnergyLevel,
                         AnalysisStrategy, LoudnessStrategy, PeakStrategy,
                         SpectralBandStrategy, PEAK_FLOOR_DBFS,
                         LOUDNESS_FLOOR_LUFS

Run with:
  pytest tests/test_audio_analysis.py -v
"""

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports from the module that does not yet exist.
# These WILL raise ImportError until the application code is written.
# That is the correct TDD red state.
# ---------------------------------------------------------------------------
from src.audio_analysis import (
    AudioAnalysisEngine,
    AudioMetrics,
    EnergyLevel,
    AnalysisStrategy,
    LoudnessStrategy,
    PeakStrategy,
    SpectralBandStrategy,
    PEAK_FLOOR_DBFS,
    LOUDNESS_FLOOR_LUFS,
)


# ===========================================================================
# Deterministic test-signal helpers
#
# All signals are bin-aligned (integer cycles per buffer at fs=48000, N=48000)
# so the FFT has zero spectral leakage and every assertion is exact/stable.
# ===========================================================================

SAMPLE_RATE = 48_000
DURATION_S = 1.0
DECIBEL_HALVING = 20.0 * math.log10(2.0)  # 6.0206 dB — exact per amplitude halving


def tone(freq_hz: float, amplitude: float = 0.5,
         dur_s: float = DURATION_S, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a mono sine tone as float64 samples in [-amplitude, amplitude]."""
    n = int(round(dur_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float64)


def silence(dur_s: float = DURATION_S, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a buffer of digital silence."""
    return np.zeros(int(round(dur_s * fs)), dtype=np.float64)


# ===========================================================================
# SECTION 1 — AnalysisStrategy abstraction (Strategy Pattern contract)
# ===========================================================================

class TestAnalysisStrategyContract:
    """The Strategy Pattern interface must be a non-instantiable ABC and all
    concrete strategies must conform to it."""

    def test_analysis_strategy_is_abstract(self):
        """AnalysisStrategy is an interface and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AnalysisStrategy()  # type: ignore[abstract]

    def test_concrete_strategies_are_analysis_strategies(self):
        """Every shipped strategy must implement the AnalysisStrategy interface."""
        for strategy in (LoudnessStrategy(), PeakStrategy(), SpectralBandStrategy()):
            assert isinstance(strategy, AnalysisStrategy)

    def test_concrete_strategies_expose_a_name(self):
        """Each strategy advertises a non-empty name for logging/identification."""
        for strategy in (LoudnessStrategy(), PeakStrategy(), SpectralBandStrategy()):
            assert isinstance(strategy.name, str) and strategy.name != ""

    def test_strategy_analyze_returns_a_dict(self):
        """A strategy's analyze() must return a dict of metric fragments."""
        result = PeakStrategy().analyze(tone(1_000.0), SAMPLE_RATE)
        assert isinstance(result, dict)


# ===========================================================================
# SECTION 2 — PeakStrategy  (Spec: "Detect peaks using numpy/scipy")
# ===========================================================================

class TestPeakStrategy:
    """Peak level is reported in dBFS: 0 dBFS == full scale (|sample| == 1.0)."""

    def test_full_scale_tone_peaks_near_zero_dbfs(self):
        result = PeakStrategy().analyze(tone(1_000.0, amplitude=1.0), SAMPLE_RATE)
        assert result["peak_level"] == pytest.approx(0.0, abs=0.1)

    def test_half_scale_tone_peaks_near_minus_six_dbfs(self):
        result = PeakStrategy().analyze(tone(1_000.0, amplitude=0.5), SAMPLE_RATE)
        assert result["peak_level"] == pytest.approx(-DECIBEL_HALVING, abs=0.1)

    def test_halving_amplitude_drops_peak_by_exactly_six_db(self):
        """Scaling the waveform by 0.5 must drop the peak by exactly 20*log10(2)."""
        full = PeakStrategy().analyze(tone(1_000.0, amplitude=1.0), SAMPLE_RATE)
        half = PeakStrategy().analyze(tone(1_000.0, amplitude=0.5), SAMPLE_RATE)
        assert full["peak_level"] - half["peak_level"] == pytest.approx(
            DECIBEL_HALVING, abs=1e-6
        )

    def test_silence_reports_peak_floor(self):
        """Digital silence has no peak; it must clamp to the defined floor."""
        result = PeakStrategy().analyze(silence(), SAMPLE_RATE)
        assert result["peak_level"] == pytest.approx(PEAK_FLOOR_DBFS)


# ===========================================================================
# SECTION 3 — LoudnessStrategy  (Spec: "Calculate LUFS loudness")
#
# Implementation is ITU-R BS.1770-4 integrated loudness (K-weighting + gating).
# ===========================================================================

class TestLoudnessStrategy:
    """Integrated loudness must be a finite, physically plausible LUFS value."""

    def test_loudness_is_in_plausible_range(self):
        result = LoudnessStrategy().analyze(tone(1_000.0, amplitude=0.5), SAMPLE_RATE)
        lufs = result["loudness_lufs"]
        assert math.isfinite(lufs)
        assert -40.0 < lufs < 0.0

    def test_halving_amplitude_drops_loudness_by_exactly_six_db(self):
        """LUFS scales with 20*log10(amplitude): halving == -6.02 LUFS, exactly.

        This holds regardless of K-weighting because scaling the input scales
        every K-weighted block's mean-square identically.
        """
        loud = LoudnessStrategy()
        full = loud.analyze(tone(1_000.0, amplitude=0.8), SAMPLE_RATE)["loudness_lufs"]
        half = loud.analyze(tone(1_000.0, amplitude=0.4), SAMPLE_RATE)["loudness_lufs"]
        assert full - half == pytest.approx(DECIBEL_HALVING, abs=1e-3)

    def test_louder_signal_has_higher_lufs(self):
        loud = LoudnessStrategy()
        quiet = loud.analyze(tone(1_000.0, amplitude=0.1), SAMPLE_RATE)["loudness_lufs"]
        loudr = loud.analyze(tone(1_000.0, amplitude=0.7), SAMPLE_RATE)["loudness_lufs"]
        assert loudr > quiet

    def test_silence_reports_loudness_floor(self):
        result = LoudnessStrategy().analyze(silence(), SAMPLE_RATE)
        assert result["loudness_lufs"] <= -70.0


# ===========================================================================
# SECTION 4 — SpectralBandStrategy
#   (Spec: FFT frequency analysis -> low_mid_energy / high_freq_energy)
# ===========================================================================

class TestSpectralBandStrategy:
    """Frequency-band energy is categorised as low / normal / high."""

    def test_low_mid_tone_is_high_in_low_mid_and_low_in_high(self):
        """A 500 Hz tone sits in the low-mid band; the high band is near-empty."""
        result = SpectralBandStrategy().analyze(tone(500.0), SAMPLE_RATE)
        assert result["low_mid_energy"] == EnergyLevel.HIGH
        assert result["high_freq_energy"] == EnergyLevel.LOW

    def test_high_freq_tone_is_high_in_high_and_low_in_low_mid(self):
        """A 12 kHz tone sits in the high band; the low-mid band is near-empty."""
        result = SpectralBandStrategy().analyze(tone(12_000.0), SAMPLE_RATE)
        assert result["high_freq_energy"] == EnergyLevel.HIGH
        assert result["low_mid_energy"] == EnergyLevel.LOW

    def test_balanced_mix_classifies_low_mid_as_normal(self):
        """Mix engineered so the low-mid band holds ~25% of total energy.

        energy ∝ amplitude², so a1²/(a1²+a2²) = 0.2²/(0.2² + (0.2√3)²) = 0.25,
        which lands strictly between the LOW (<0.10) and HIGH (>=0.40) gates.
        """
        a1 = 0.2
        a2 = 0.2 * math.sqrt(3.0)  # gives exactly a 1:3 energy split
        mix = tone(500.0, amplitude=a1) + tone(12_000.0, amplitude=a2)
        result = SpectralBandStrategy().analyze(mix, SAMPLE_RATE)
        assert result["low_mid_energy"] == EnergyLevel.NORMAL
        assert result["high_freq_energy"] == EnergyLevel.HIGH

    def test_band_classification_is_returned_as_energy_level_enum(self):
        result = SpectralBandStrategy().analyze(tone(500.0), SAMPLE_RATE)
        assert isinstance(result["low_mid_energy"], EnergyLevel)
        assert isinstance(result["high_freq_energy"], EnergyLevel)


# ===========================================================================
# SECTION 5 — AudioMetrics structured output (Pydantic validation)
# ===========================================================================

class TestAudioMetricsModel:
    """The engine's output is a validated, structured object (Pydantic)."""

    def test_audio_metrics_accepts_valid_payload(self):
        metrics = AudioMetrics(
            loudness_lufs=-15.0,
            peak_level=-1.2,
            low_mid_energy="high",
            high_freq_energy="normal",
        )
        assert metrics.loudness_lufs == -15.0
        assert metrics.peak_level == -1.2
        assert metrics.low_mid_energy == EnergyLevel.HIGH
        assert metrics.high_freq_energy == EnergyLevel.NORMAL

    def test_audio_metrics_rejects_invalid_energy_level(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AudioMetrics(
                loudness_lufs=-15.0,
                peak_level=-1.2,
                low_mid_energy="enormous",  # not a valid EnergyLevel
                high_freq_energy="normal",
            )

    def test_audio_metrics_matches_spec_example_shape(self):
        """The model exposes exactly the four fields from the spec example."""
        metrics = AudioMetrics(
            loudness_lufs=-15.0,
            peak_level=-1.2,
            low_mid_energy="high",
            high_freq_energy="normal",
        )
        dumped = metrics.model_dump()
        assert set(dumped.keys()) == {
            "loudness_lufs",
            "peak_level",
            "low_mid_energy",
            "high_freq_energy",
        }


# ===========================================================================
# SECTION 6 — AudioAnalysisEngine (aggregation + Strategy Pattern pluggability)
# ===========================================================================

class TestAudioAnalysisEngine:
    """The engine runs its strategies and aggregates a validated AudioMetrics."""

    def test_default_engine_returns_audio_metrics(self):
        engine = AudioAnalysisEngine.default()
        metrics = engine.analyze(tone(1_000.0, amplitude=0.5), SAMPLE_RATE)
        assert isinstance(metrics, AudioMetrics)

    def test_default_engine_produces_all_four_metrics(self):
        """End-to-end on a known low-mid tone: every spec field is populated
        and self-consistent with the individual strategies."""
        engine = AudioAnalysisEngine.default()
        metrics = engine.analyze(tone(500.0, amplitude=0.3), SAMPLE_RATE)

        # Peak of a 0.3-amplitude tone ~= 20*log10(0.3) = -10.46 dBFS
        assert metrics.peak_level == pytest.approx(20.0 * math.log10(0.3), abs=0.1)
        assert math.isfinite(metrics.loudness_lufs)
        assert metrics.low_mid_energy == EnergyLevel.HIGH
        assert metrics.high_freq_energy == EnergyLevel.LOW

    def test_default_engine_has_three_strategies(self):
        engine = AudioAnalysisEngine.default()
        assert len(engine.strategies) == 3

    def test_empty_buffer_raises_value_error(self):
        engine = AudioAnalysisEngine.default()
        with pytest.raises(ValueError):
            engine.analyze(np.array([], dtype=np.float64), SAMPLE_RATE)

    def test_non_finite_buffer_raises_value_error(self):
        engine = AudioAnalysisEngine.default()
        bad = tone(1_000.0)
        bad[100] = np.nan
        with pytest.raises(ValueError):
            engine.analyze(bad, SAMPLE_RATE)

    def test_stereo_buffer_is_downmixed_to_mono(self):
        """A 2-D (frames, channels) buffer is averaged to mono without error."""
        mono = tone(1_000.0, amplitude=0.5)
        stereo = np.stack([mono, mono], axis=1)  # shape (N, 2)
        engine = AudioAnalysisEngine.default()
        metrics = engine.analyze(stereo, SAMPLE_RATE)
        # Identical channels => downmix equals the mono signal => same peak.
        mono_metrics = engine.analyze(mono, SAMPLE_RATE)
        assert metrics.peak_level == pytest.approx(mono_metrics.peak_level, abs=1e-9)

    # ------------------------------------------------------------------
    # Strategy Pattern: a custom strategy can be swapped in transparently
    # ------------------------------------------------------------------

    def test_custom_strategy_can_be_plugged_into_engine(self):
        """The engine depends only on the AnalysisStrategy abstraction, so a
        fake loudness algorithm can replace the real one with no engine change.
        This is the core Strategy Pattern guarantee."""

        class FixedLoudnessStrategy(AnalysisStrategy):
            @property
            def name(self) -> str:
                return "fixed-loudness"

            def analyze(self, samples, sample_rate):
                return {"loudness_lufs": -14.0}

        engine = AudioAnalysisEngine(
            [FixedLoudnessStrategy(), PeakStrategy(), SpectralBandStrategy()]
        )
        metrics = engine.analyze(tone(500.0, amplitude=0.5), SAMPLE_RATE)

        # The injected strategy fully determines the loudness field.
        assert metrics.loudness_lufs == -14.0
        # The other (real) strategies still contribute their fields.
        assert metrics.low_mid_energy == EnergyLevel.HIGH

    def test_later_strategy_overrides_earlier_field(self):
        """Strategies are applied in order; a later strategy may override an
        earlier strategy's field — demonstrating ordered composition."""

        class OverridePeakStrategy(AnalysisStrategy):
            @property
            def name(self) -> str:
                return "override-peak"

            def analyze(self, samples, sample_rate):
                return {"peak_level": -3.3}

        engine = AudioAnalysisEngine(
            [
                PeakStrategy(),            # produces the real peak first
                OverridePeakStrategy(),   # then overrides it
                LoudnessStrategy(),
                SpectralBandStrategy(),
            ]
        )
        metrics = engine.analyze(tone(500.0, amplitude=0.5), SAMPLE_RATE)
        assert metrics.peak_level == -3.3
