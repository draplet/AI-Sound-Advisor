# Fake X32 Simulator - Project Specification

**Goal**: Create a lightweight X32 impersonator that runs on a local network to allow Audio Pilot to test real networking without hardware access.

**Status**: Specification Phase

---

## Project Overview

**Fake X32** is a standalone web-based OSC server that:
- Impersonates a Behringer X32 mixer over UDP/OSC on port 10023
- Runs on a local network (different machine from Audio Pilot)
- Provides 32 simulated channels, each with toggleable problem states
- Allows Audio Pilot to test real network connectivity and data processing

**When complete, Audio Pilot should**:
1. Point to Fake X32's IP address instead of a real mixer
2. Connect successfully via `/info` handshake
3. Query channel data and receive realistic responses
4. Display metrics based on simulated problem states
5. Show "Your mix is sounding good" when all channels except one are toggled off

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Machine A: Fake X32 Simulator                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ Web UI (localhost:5000)                      │   │
│  │ - 32 channel toggles                         │   │
│  │ - Problem state selector per channel         │   │
│  │ - Live metrics display                       │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │ OSC Server (UDP port 10023)                  │   │
│  │ - Responds to /info queries                  │   │
│  │ - Responds to channel state queries          │   │
│  │ - Handles /xremote keepalive                 │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │ Metrics Engine                               │   │
│  │ - Generates LUFS, peaks, frequencies         │   │
│  │ - Applies problem state modifiers            │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
          ↕ (UDP OSC on port 10023)
        Network
          ↕
┌─────────────────────────────────────────────────────┐
│  Machine B: Audio Pilot                             │
│  - Connects to Fake X32 IP:10023                    │
│  - Receives /info response                          │
│  - Queries channel state                            │
│  - Processes metrics and detects issues             │
└─────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. **OSC Server** (`fake_x32_osc.py`)
   - Listen on UDP port 10023
   - Handle incoming OSC messages:
     - `/info` → return `["V2.07", "fake-x32-simulator", "X32", "4.06"]`
     - `/ch/NN/config/name` → return channel name (e.g., "Lead Vocal")
     - `/ch/NN/mix/fader` → return fader level (0.0-1.0)
     - `/ch/NN/mix/on` → return mute state (0 or 1)
     - `/xremote` → acknowledge keepalive
   - Return realistic values based on current channel state

### 2. **Metrics Engine** (`fake_x32_metrics.py`)
   - Maintain 3 simulated channels + 1 recorder channel
   - Each channel can be in one of 3 states:
     - **Good Levels** (default, clean)
     - **Clipping** (hot peaks, risk of distortion)
     - **Muddy Vocals** (high low-mid energy, buried clarity)
   - Generate metrics (LUFS, peak level, frequency distribution) based on state
   - Calculate master LUFS from all active channels

### 3. **Simulated Mixer State** (`fake_x32_mixer.py`)
   - **3 Channels**:
     - Channel 1: "Lead Vocal"
     - Channel 2: "Drums"
     - Channel 3: "Bass"
   - **Recorder Channel (33)**: Mix of all active channels
   - Track each channel's:
     - Name
     - Current problem state
     - Derived metrics (LUFS, peaks, frequencies)
   - Update metrics in real-time when state changes via CLI

---

## Channel Configuration (3 Channels + Recorder)

**Simulated Channels:**
- Channel 1: "Lead Vocal"
- Channel 2: "Drums"
- Channel 3: "Bass"
- Channel 33 (Recorder): Mix of all active channels above

---

## Problem State Definitions

Each channel can be toggled into one of these 3 states. The metrics engine returns hardcoded values:

| State | LUFS | Peak Level | Low-Mid Energy | High Freq | Description |
|-------|------|-----------|---|---|---|
| **Good Levels** | -22 | -10 dB | NORMAL | NORMAL | Clean, well-balanced |
| **Clipping** | -12 | 0.0 dB | NORMAL | NORMAL | Hot peaks, risk of distortion |
| **Muddy Vocals** | -20 | -8 dB | HIGH | LOW | Heavy low-mid buildup; vocal clarity lost |

**Recorder Channel**: Always a **mix** of all active channels' metrics (averaged LUFS, summed peaks, etc.).

---

## OSC Message Specification

Audio Pilot queries Fake X32 with these OSC messages. Responses must match format:

### Query: `/info`
**Response**: `["V2.07", "fake-x32-simulator", "X32", "4.06"]`
- Used by Audio Pilot on initial connect
- Fake X32 identifies itself

### Query: `/ch/{01-03}/config/name`
**Response**: String (channel name from: "Lead Vocal", "Drums", "Bass")
- Audio Pilot requests channel labels
- Return preloaded name for channels 1-3

### Query: `/ch/{01-03}/mix/fader`
**Response**: Float 0.0–1.0 (representing fader position)
- 0.5 = unity / 0 dB
- < 0.5 = attenuated
- > 0.5 = boosted

### Query: `/ch/{01-03}/mix/on`
**Response**: Int 0 or 1
- 1 = channel active/unmuted
- 0 = muted/inactive
- Tied to channel's on/off toggle via CLI

### Query: `/xremote`
**Response**: Acknowledgement (or silent)
- Keepalive message sent by Audio Pilot every ~9 seconds
- Fake X32 just acknowledges; connection stays alive

---

## Implementation Priorities

### **Phase 1: Foundation** ⭐ START HERE
- [ ] Create OSC server skeleton (listen on UDP 10023)
- [ ] Implement `/info` response (identify as "Fake x32")
- [ ] Implement `/xremote` keepalive handling
- [ ] Test: Audio Pilot can connect and see "Connected to: Fake x32 Firmware 4.06"
- **Deliverable**: Proof of network connection

### **Phase 2: Mixer State (3 Channels)**
- [ ] Implement 3-channel mixer state model (Lead Vocal, Drums, Bass)
- [ ] Preload channel names
- [ ] Implement `/ch/NN/config/name` queries
- [ ] Implement `/ch/NN/mix/fader` queries
- [ ] Implement `/ch/NN/mix/on` queries (mute state)
- **Deliverable**: Audio Pilot can read channel data from Fake X32

### **Phase 3: Metrics Engine (Hardcoded Values + CLI)**
- [ ] Create 3 problem state definitions (Good Levels, Clipping, Muddy Vocals)
- [ ] Implement hardcoded LUFS values per state
- [ ] Implement hardcoded peak levels per state
- [ ] Implement hardcoded frequency energy per state
- [ ] Wire metrics to OSC responses
- [ ] Create CLI interface to toggle channels and states
- [ ] Display current state in console
- **Deliverable**: Different problem states change metrics values; operator can control via CLI

### **Phase 4: Integration & Testing** (CLI ONLY — NO WEB UI)
- [ ] Start Fake X32 script, verify IP is displayed
- [ ] Point Audio Pilot to Fake X32 IP:10023
- [ ] Verify Audio Pilot connects successfully
- [ ] Toggle channels via CLI commands (e.g., `toggle 1 clipping`)
- [ ] Verify Audio Pilot detects metric changes and shows alerts
- [ ] Test "all off except one" → should see "Your mix is sounding good"
- **Deliverable**: Full end-to-end network testing workflow

---

## Technical Stack

- **Language**: Python 3.8+
- **OSC**: `python-osc` library
- **Web UI**: Flask or FastAPI + HTML/CSS/JavaScript
- **Testing**: Manual via Audio Pilot + Web UI

---

## File Structure

```
fake_x32/
├── fake_x32_osc.py           # OSC server, message handling
├── fake_x32_metrics.py       # Metrics engine, problem states
├── fake_x32_mixer.py         # Mixer state model (3 channels)
├── run_fake_x32.py           # Entry point; CLI interface + OSC server
├── config.py                 # Channel names, default state values
└── README.md                 # Setup and usage instructions
```

---

## Success Criteria

✅ **Phase 1 Complete**: Audio Pilot connects to Fake X32 over network and displays "Connected to: Fake x32 Firmware 4.06"

✅ **Phase 2 Complete**: Audio Pilot can read channel names (Lead Vocal, Drums, Bass) and state from Fake X32

✅ **Phase 3 Complete**: CLI commands (toggle, set) change metrics; Audio Pilot detects and reports issues in real-time

✅ **Phase 4 Complete**: Full end-to-end testing: start both apps on network, toggle scenarios via CLI, see Audio Pilot alerts update

---

## Setup Instructions (Draft)

### On Fake X32 Machine:
```bash
python run_fake_x32.py
# Output: 
# OSC Server listening on 0.0.0.0:10023
# Your IP address is: 192.168.1.X
# 
# CLI Commands:
#   toggle 1           - Toggle channel 1 on/off
#   set 1 clipping     - Set channel 1 to "Clipping" state
#   set 1 good         - Set channel 1 to "Good Levels" state
#   set 1 muddy        - Set channel 1 to "Muddy Vocals" state
#   status             - Show current channel states
#   help               - Show all commands
```

### On Audio Pilot Machine:
1. Open Audio Pilot dashboard
2. Go to "X32 Network" settings panel
3. Enter Fake X32's IP address (shown on Fake X32 startup)
4. Port: 10023
5. Click "Connect"
6. Should see: "Connected to: Fake x32 Firmware 4.06"

### Testing a Scenario:
1. On Fake X32 CLI, type: `set 1 clipping`
2. On Audio Pilot, should see alert: "Clipping risk detected on Lead Vocal"
3. Confidence should be High or Medium
4. When you toggle off all problem channels (all set to "good"), Audio Pilot shows "Your mix is sounding good"

---

## Questions for Refinement

- Should Fake X32 display its IP address prominently on startup?
- Do you want the web UI to show connection logs (what Audio Pilot is querying)?
- Should problem states be randomizable for stress testing?
- Any other channels you'd like preloaded besides the defaults?

---

## Notes

- **No persistence**: Fake X32 doesn't save state between restarts
- **No audio streaming**: Only metrics; Audio Pilot analyzes its own local audio
- **Minimal OSC surface**: Only implement messages that Audio Pilot actually queries
- **Nameserver agnostic**: Operator manually enters IP on Audio Pilot (no mDNS/Bonjour needed for Phase 1)

