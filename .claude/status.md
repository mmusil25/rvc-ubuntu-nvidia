# RVC Realtime — Project Status

_Last updated: 2026-06-06. Goal: low-latency, realistic realtime voice conversion for Discord (mic → RVC → virtual mic). Hardware: RTX 4090 + i9-14900KS (P-cores 0–15, E-cores 16–31), Ubuntu, PipeWire._

## TL;DR — where we are
Audio quality is **dialed in and "peaking."** The choppiness that started this work is fixed. The realtime loop is stable: **0 xruns, ~27 ms infer vs ~200 ms budget, jitter locked to the block period.** Remaining work is polish + the roadmap's "advanced latency" items.

## How to run / watch
```bash
runrvc     # bash alias -> run-realtime-gui.sh, detached (sets all env below)
rvclog     # bash alias -> tail -F /tmp/rvc_debug.log  (live metrics)
```
- GUI/console log: `/tmp/rvc_gui.log`  • Metrics log: `/tmp/rvc_debug.log`
- In the GUI set **both Input and Output device = `pulse`** (not a raw `hw:` device — PipeWire owns it).
- Metrics vocabulary: `SUMMARY` (rolling 1/s), `XRUN` (device under/overflow), `OVERBUDGET` (missed deadline; the only expected one is cb#1 warmup), `CLIP`, plus `sola` (splice stability), `jitter`, per-stage `fea/index/f0/model` ms.

## Current dialed-in settings (persisted in `configs/inuse/config.json`)
- **Phase vocoder: ON** ← the key fix for choppiness. Keep it on.
- Index Rate **0.5**, rms_mix 0.83, pitch **+11**, f0 = **crepe**, threshold −60 (gate off)
- Latency: block_time **0.20**, crossfade 0.15, extra_time 2.0
- Input EQ on: high **−9 dB** (cuts mic hiss before the model). Output EQ on: low −2.5 / mid +2.5 / high −8.
- Advanced: FP16 on, NR strength 0.79, SOLA search 16 ms
- Voice model: `../voices/path/egirl.pth` + `../voices/index/egirl.index`

## What was built this session (all in repo, uncommitted on `main`)
**New modules** (`infer/lib/`):
- `rt_debug.py` — audio-callback-safe metrics logger (background writer thread → `/tmp/rvc_debug.log`).
- `rt_eq.py` — 3-band RBJ biquad EQ (low-shelf 120 / peak 1.5k / high-shelf 6k). Block-continuous via `sosfilt` `zi` (verified zero splice error → no added clicks).
- `rt_priority.py` — P-core pinning (`sched_setaffinity`) + audio-thread `SCHED_FIFO` (works w/o sudo: audio group + rtprio 99).
- `rt_waterfall.py` — rolling FFT spectrogram (log freq 50 Hz–12 kHz, adaptive dB, PIL→PNG). Replaced the old time-domain scopes.

**`gui_v1.py`** — forced English (`I18nAuto("en_US")`); caption text under every control (10 pt); Input/Output EQ frames (hot); **spectrogram waterfalls** (input top / output bottom, bottom-right under Performance); advanced frame (FP16/FP32, NR strength, SOLA search ms); debug banner + per-callback `dbg.tick`; CPU affinity at startup + FIFO boost on first callback. Event loop is now timeout-driven (`read(timeout=50)`) to animate the waterfall.
**`rtrvc.py`** — `nprobe=8` on index load (was 1 → ~19% of frames silently skipped retrieval); index warning throttled to once; per-stage timings routed to the logger.
**`run-realtime-gui.sh`** — added `RVC_DEBUG*`, `RVC_CPU_AFFINITY`, `RVC_RT_PRIO` env.

## Threading model (important)
- **Audio thread** (SCHED_FIFO, P-core): the only place real work runs. For the waterfall it just stashes the latest block by reference + bumps `_wf_seq`.
- **GUI thread** (P-core, pinned with the process): FFT + colormap + PNG encode for the waterfall (~1.7 ms/block), only when a new block arrives.

## Key diagnostic findings (don't relitigate these)
- Choppiness was **SOLA splice-offset instability**, NOT device buffering / GPU / clipping (all measured clean). Fix = phase vocoder ON (+ Index Rate ~0.5). `sola` collapses toward 0 when healthy; it pegged the search ceiling when broken.
- The "Invalid index / use added not trained" spam was a **misdiagnosis by upstream** — the index is valid; root cause was `nprobe=1` hitting sparse IVF cells. Fixed.
- 4090 has massive headroom (~27 ms infer); CPU scheduling / E-core migration was the real jitter risk → fixed with affinity + FIFO.

## Roadmap status
1. ✅ Stabilize audio at current feature set
2. ✅ Realtime visualizers (now frequency-domain waterfalls)
3. ✅ Input/output equalizers
4. ⏳ Advanced latency work — **next up**

## Open items / next steps
- **Commit the work** — still uncommitted on `main`. 4 modified files + 4 new modules. (Mark hasn't asked to commit yet.)
- Waterfall niceties Mark may want: frequency-axis tick labels (100/1k/10k), fixed-dB-scale toggle for cross-session comparison.
- Roadmap #4 latency ideas: push block_time toward 0.10 (budget headroom is huge) while watching `OVERBUDGET`/`xruns`; try smaller crossfade; experiment with f0 method (rmvpe vs crepe) for latency/quality; consider a different sampling window.
- `RVC_DEBUG_SYNC=1` for one run gives the true per-stage GPU split (default timings are launch-skewed).

## Gotchas
- **Line endings: `gui_v1.py` and `rtrvc.py` are CRLF.** When editing `rtrvc.py` via Python, write with `open(p,'w',newline='\r\n')` or you'll flip the whole file to LF and create a 900-line phantom diff. (The Edit tool preserves CRLF fine.)
- New GUI controls that shouldn't stop the stream must be handled explicitly in the event loop before the `event != "start_vc"` catch-all (which calls `stop_stream`). EQ + waterfall are handled; advanced sliders use `enable_events=False`.
- Env knobs: `RVC_RT_PRIO` (FIFO prio, 0=off, keep < PipeWire ~88), `RVC_CPU_AFFINITY` (1/0), `RVC_DEBUG` / `RVC_DEBUG_LOG` / `RVC_DEBUG_INTERVAL` / `RVC_DEBUG_SYNC`.
