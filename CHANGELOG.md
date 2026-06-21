# Changelog

Dated log of notable changes to the AI Sound Advisor. Newest entries first.

---

## 2026-06-20

Real audio analysis in the dashboard, clearer per-suggestion channel labels, and
project documentation (dependencies + this changelog).

### Real audio analysis in run_dashboard
- `run_dashboard` now analyzes a REAL audio input by default (was a simulated
  feed), so readings and clipping/loudness/EQ suggestions reflect the live
  signal. Defaults to the Windows default input (room mic); selectable via
  `SOUND_ADVISOR_AUDIO_DEVICE` (index or name substring, e.g. "X32"), with
  `SOUND_ADVISOR_SAMPLE_RATE` / `_AUDIO_CHANNELS`. Set
  `SOUND_ADVISOR_USE_REAL_AUDIO=0` to fall back to the simulated feed.
- Failsafe: if the device can't be read it degrades to "audio lost"/waiting and
  never claims the mix is good.
- Note: the X32 must reach this PC as an audio device (e.g. over USB) for the
  actual board feed; otherwise only the room mic is available. Detection is
  still mix-wide, so most suggestions are about the Main Mix (see below).
- Files: `run_dashboard.py`.

### Per-suggestion channel labels
- Every suggestion now leads with its target: "Channel 03, Lead Vocal — ..."
  for channel-specific issues (vocal masking) or "Main Mix — ..." for mix-wide
  issues (clipping/loudness/EQ). Replaces the earlier `/ch/01/"..."` style.
- Files: `src/suggestion.py` (`channel_ref`/`render`), `src/templates/index.html`,
  `tests/test_suggestion.py`.
- Backlog: real per-channel attribution for clipping/loudness/EQ needs
  per-channel audio + real mixer state (see "Bug fixes i want completed").

### Project documentation

### Dependency export (for packaging)
- Added `requirements-lock.txt` — a full `pip freeze` of every package
  (including transitive deps) at exact versions, captured from the verified
  Python 3.10.11 environment. Feed to pip to rebuild the exact environment for
  an install package; regenerate whenever dependencies change.
- Added `DEPENDENCIES.md` — human-readable overview of dependencies grouped by
  role (runtime / real-hardware extras / AI-LLM / dev-build-test), with versions
  and what each is for. Notes that `librosa`/`pyloudnorm`/`scipy` from the plan
  are not used (DSP is pure NumPy) and that the AI needs Ollama + a model
  installed outside pip.

### Changelog
- Added `CHANGELOG.md` (this file) as a dated log of notable changes.

---

## 2026-06-19

Focused work on the `run_dashboard` launcher: making the X32 networking real,
fixing how it is launched, making the on-screen status honest, enabling the
local AI, and improving how suggestions are displayed.

### X32 network connection (real, over UDP/OSC)
- `run_dashboard` previously used an in-memory fake mixer, so the Settings
  "Test Connection" button always reported success regardless of the IP.
- Rewired the Connect panel and Test Connection to a real `UdpOscChannel`
  using the saved `config.json` settings. `/info` is now actually sent over the
  network: success (with the console's model/firmware) is shown only on a
  genuine reply; a wrong IP / missing mixer / no network times out and is shown
  as an error.
- Files: `run_dashboard.py`, `src/x32_connection.py` (existing logic confirmed).

### Launcher fixes (`run_dashboard.bat` + requirements)
- Root cause of "No module named 'uvicorn'": the machine has Python 3.10, 3.12
  and 3.14 installed, but only 3.10 has the packages; double-clicking the `.py`
  used 3.14.
- Added `run_dashboard.bat` that forces `py -3.10`, checks/installs
  dependencies, and starts the dashboard.
- Added `requirements.txt` pinned to the verified-working 3.10 versions.
- The launcher frees port 8001 before starting (avoids the "[Errno 10048] only
  one usage of each socket address" error from a lingering previous instance).
- The launcher ensures the local Ollama server is running before launch (probes
  the API, starts `ollama serve` and waits if needed); failsafe if Ollama is
  absent.

### Honest status banner (no false "sounding good")
- The dashboard no longer claims "Your mix is sounding good" before anything is
  connected. Until the master X32 connection is up (and audio is present) it
  shows a "Waiting for communication — connect the X32 to begin." banner with no
  issues/suggestions. The all-clear message only appears once a good mix is
  actually measured.
- Stopped the rolling demo by default (steady clean feed, no fake recording
  mismatch); set `SIMULATE_ISSUES = True` in `run_dashboard.py` to replay the
  old clipping/masking cycle.
- Files: `run_dashboard.py` (`WaitingAwareOrchestrator`), `src/templates/index.html`.

### Local AI (Ollama) in the dashboard
- `run_dashboard` now wires the local Ollama LLM into the AI panel
  (default model `qwen2.5:1.5b`, overridable via `SOUND_ADVISOR_LLM_*` env
  vars). The AI/LLM status indicator reports available, and chat replies come
  from the local model. Failsafe: silently falls back to built-in basic alerts
  if Ollama is down or the model is missing.
- Files: `run_dashboard.py`.

### Suggestion display — OSC channel tags
- Channel-specific suggestions now lead with an OSC-style tag, e.g.
  `/ch/01/"Lead Vocal"`, followed by the message. Main-mix issues (clipping,
  loudness, EQ) show no tag.
- Threaded the 1-based channel number from detection through to the UI:
  added `Issue.channel_index`, `Suggestion.channel_index` + a `channel_ref`
  formatter, a `channel_ref` payload field, and a monospace prefix in the
  suggestions panel. Tests added/updated.
- Files: `src/detection.py`, `src/suggestion.py`, `src/web_server.py`,
  `src/templates/index.html`, `tests/test_detection.py`, `tests/test_suggestion.py`.

### Tests
- Full suite green: 390 passing (the only failure is the unrelated, untracked
  `tests/test_fake_x32_osc.py` work-in-progress).

### Commits
- `4743812` — Wire run_dashboard Connect/Test to the real X32 over UDP/OSC
- `18e05c5` — Add Python 3.10 launcher + pinned requirements for the dashboard
- `ba91194` — Dashboard: honest status banner, Ollama AI, and OSC channel tags
