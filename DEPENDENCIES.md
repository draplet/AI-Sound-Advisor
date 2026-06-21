# Dependencies

Reference list of what the AI Sound Advisor needs to run and to package.
**Update this whenever dependencies change** (and regenerate `requirements-lock.txt`).

- **Last updated:** 2026-06-20
- **Python:** 3.10.11 (the project is pinned to Python 3.10 — see `run_dashboard.bat`)
- **Files:**
  - `requirements.txt` — minimal, hand-maintained runtime list (what you edit).
  - `requirements-lock.txt` — full `pip freeze` of every package incl. transitive
    deps, exact versions — feed this to pip to rebuild the exact environment for
    an install package.
  - `DEPENDENCIES.md` — this document (the readable overview).

Install the locked environment:

```
py -3.10 -m pip install -r requirements-lock.txt
```

---

## Runtime — required (the dashboard / web server)

| Package | Version | Purpose |
|---|---|---|
| fastapi | 0.136.3 | Web API + dashboard backend |
| uvicorn | 0.49.0 | ASGI server that runs the app |
| websockets | 16.0 | Real-time WebSocket push to the dashboard |
| pydantic | 2.13.4 | Data models + validation (all agent I/O) |
| numpy | 2.2.6 | Audio DSP — LUFS, FFT bands, peak detection |
| python-osc | 1.10.2 | OSC/UDP communication with the Behringer X32 |

Transitive deps these pull in (pinned in `requirements-lock.txt`): starlette,
anyio, h11, click, colorama, idna, annotated-types, typing_extensions,
typing-inspection, exceptiongroup, tomli, pydantic_core.

## Real-hardware extras — optional

Only needed for the live build (`build_default_app` / `run_real_test.bat`); the
hardware-free `run_dashboard` demo does not require them.

| Package | Version | Purpose |
|---|---|---|
| sounddevice | 0.5.5 | Live microphone / audio-interface capture |
| webrtcvad | 2.0.10 | Voice-activity detection (optional spec extension) |

Pulls in: cffi, pycparser (sounddevice's CFFI backend).

> **Note:** the master plan names `librosa`, `pyloudnorm` and `scipy`, but the
> DSP is implemented in pure NumPy (a self-contained BS.1770-4 loudness meter +
> FFT). Those three are **not installed** and not required.

## AI / LLM — no pip dependency

The LLM client talks to Ollama / LM Studio over HTTP using only the Python
**standard library** (`urllib`), so there is no Python package to install.

External (non-pip) requirements for the AI panel:
- **Ollama** desktop app/server installed and running.
- A pulled model, default **`qwen2.5:1.5b`** (`ollama pull qwen2.5:1.5b`).

## Dev / test / build — not shipped to end users

| Package | Version | Purpose |
|---|---|---|
| pytest | 9.0.3 | Test runner |
| httpx | 0.28.1 | FastAPI TestClient (tests) |
| build | 1.5.0 | Build wheels/sdists for packaging |
| pip-tools | 7.5.3 | Compile/pin requirements |

Pulls in: pluggy, iniconfig, Pygments, packaging, pyproject_hooks, httpcore,
certifi.

---

## How to regenerate this list

```
py -3.10 -m pip freeze > requirements-lock.txt
```

Then review the version table above and bump anything that changed. Also note
the external (non-pip) versions when they move: Python itself, Ollama, and the
model tag.
