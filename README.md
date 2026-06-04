# RVC Realtime Voice Changer — Ubuntu + NVIDIA setup notes

This is a working configuration of [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)'s **realtime GUI** (`gui_v1.py`) for live voice conversion into **Discord**, set up on one specific machine. It is intentionally machine-specific — device names, card IDs, and the GPU are hard-wired to this box.

> The original upstream project README is preserved as [`README-upstream.md`](README-upstream.md).

## This machine

| | |
|---|---|
| OS | Ubuntu (kernel 6.17), X11 session |
| Audio server | **PipeWire 1.0.5** (PulseAudio compat via `pactl`) |
| GPU | **NVIDIA GeForce RTX 4090** (24 GB), driver 595.71.05 (CUDA 13.2 capable) |
| Audio interface | **M-Audio M-TRACK DUO HD** (USB) — analog mic/instrument inputs |
| Denoise | **NoiseTorch** virtual mic on top of the M-TRACK |
| Discord | installed as a **Snap** (`/snap/bin/discord`) |

## TL;DR — how to run it

```bash
./run-realtime-gui.sh
```

Then in the GUI: **Sound API = ALSA**, **Input Device = `pulse`**, **Output Device = `pulse`**, load your model + index, click **Start Audio Conversion**.
In Discord: **Input Device = `RVC_Virtual_Mic`** (fully quit + reopen Discord first so the Snap re-scans devices).

Audio chain:

```
M-TRACK mic → NoiseTorch (denoise) → RVC GUI [input: pulse]
            → RTX 4090 voice conversion (fp16)
            → RVC GUI [output: pulse] → RVC_to_Discord (null sink) → RVC_Virtual_Mic → Discord
```

---

## What was done to get here

### 1. Python environment — must be 3.10 (not 3.11/3.12)

The realtime stack depends on **fairseq 0.12.2**, which only works on **Python 3.8–3.10**. On 3.11+ you hit a hard wall: Python 3.11 added a dataclass guard (`__hash__ is None` → "mutable default … use default_factory") that rejects fairseq/hydra's `field = SomeConfig()` defaults; converting them to `default_factory` then breaks **omegaconf 2.0.6**, which fairseq pins (`<2.1`) and which doesn't understand `default_factory`. Dead end.

This machine only had system Python 3.11/3.12, so [`uv`](https://github.com/astral-sh/uv) was used to get a standalone 3.10 without `sudo`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv venv --python 3.10 .venv      # the old 3.11 venv was moved to .venv_py311_bak
```

`.venv/` is git-ignored (~7.6 GB). Recreate it with the steps below.

### 2. Dependencies

```bash
export VIRTUAL_ENV="$PWD/.venv"
# CUDA build of torch (driver 595 is backward-compatible with cu12.8)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# realtime GUI stack
uv pip install "setuptools<81" wheel Cython \
  einops flask flask_cors gin gin_config librosa local_attention matplotlib \
  praat-parselmouth pyworld PyYAML resampy scikit_learn scipy SoundFile \
  tensorboard tqdm FreeSimpleGUI gradio noisereduce torchfcpe \
  "sounddevice<0.5.0" faiss-cpu torchcrepe python-dotenv
```

Gotchas in the upstream `requirements-win-for-realtime_vc_gui.txt`:
- **`wave`** — a bogus abandoned PyPI package that pulls `MySQL-python` and aborts the whole `pip` batch. Python already has `wave` built in. **Omit it.**
- **`PySimpleGUI`** — the code actually imports **`FreeSimpleGUI`** (the open fork). Install that instead.

### 3. fairseq (built from source + patched for the toolchain)

fairseq has no usable wheel here; the PyPI sdist build fails on `version.txt`. Build from a git checkout:

```bash
git clone --depth 1 https://github.com/facebookresearch/fairseq.git /tmp/fairseq-src
uv pip install /tmp/fairseq-src --no-build-isolation
```

`omegaconf 2.0.6` ships invalid wheel metadata that modern pip rejects; `uv` (or `pip<24.1`) installs it fine.

### 4. Runtime patches

- **`infer/lib/rtrvc.py`** (committed): registers `fairseq.data.dictionary.Dictionary` as a torch "safe global" so HuBERT loads under PyTorch ≥2.6's `weights_only` default:
  ```python
  from fairseq.data.dictionary import Dictionary
  torch.serialization.add_safe_globals([Dictionary])
  ```
- **`gui_v1.py`** (committed): the host-API change handler crashed with `IndexError` when an empty host API (OSS) was selected — the output-device branch was missing the `len(...) > 0` guard the input branch had. Fixed so picking an empty API no longer kills the GUI.
- **fairseq/hydra dataclass fixes** — *not committed* (they live in the git-ignored `.venv`). Only needed if you ever run on Python 3.11+; on 3.10 they are unnecessary. For reference, the changes were: in `fairseq/dataclass/configs.py` (`FairseqConfig`) and `hydra/conf/__init__.py`, rewrite `name: T = T()` defaults as `name: T = field(default_factory=T)`.

### 5. Audio routing (PipeWire) — done by `run-realtime-gui.sh`

Why the GUI never showed the mic by name: PortAudio (what `sounddevice`/the GUI uses) only exposes the **ALSA** host API here, and it can't open the M-TRACK directly because **PipeWire holds it** (and NoiseTorch + Discord were recording from it). So the M-TRACK never appears as a named input. The fix is to go through PortAudio's **`pulse`** device and pin routing with env vars.

The launch script does, each run:

1. **Forces the M-TRACK to its analog profile** — it defaults to the digital/IEC958 profile, which leaves the analog mic/instrument inputs dead:
   ```bash
   pactl set-card-profile alsa_card.usb-M-Audio_M-TRACK_DUO_HD_5000000001-01 \
         output:analog-stereo+input:analog-stereo
   ```
2. **Creates the virtual cable** (RVC's output target):
   ```bash
   pactl load-module module-null-sink sink_name=RVC_to_Discord \
         sink_properties=device.description=RVC_to_Discord
   ```
3. **Exposes the cable as a real microphone** so the sandboxed Snap Discord lists it (Discord hides plain `*.monitor` sources):
   ```bash
   pactl load-module module-remap-source master=RVC_to_Discord.monitor \
         source_name=RVC_Virtual_Mic source_properties=device.description=RVC_Virtual_Mic
   ```
4. **Pins the GUI's PulseAudio streams** — record from the NoiseTorch denoised mic, play out to the cable:
   ```bash
   export PULSE_SOURCE="NoiseTorch Microphone for M-TRACK DUO HD"   # falls back to raw M-TRACK
   export PULSE_SINK=RVC_to_Discord
   ```

The null sink and remap source are **not persistent across reboot**; the script recreates them each run.

### 6. Discord (Snap)

Snap Discord caches its device list at startup and hides monitor sources, so after the virtual mic exists you must **fully quit Discord (Ctrl+Q) and reopen it**, then set **Input Device = `RVC_Virtual_Mic`** and disable Discord's own Noise Suppression / Echo Cancellation / AGC (NoiseTorch + RVC already condition the audio). The `audio-record` Snap interface is connected (`snap connections discord`).

## Notes / tuning

- RVC runs on the 4090 in **fp16** (`device: cuda:0`). Lower the GUI **block time** (~0.10–0.15 s) for tighter latency; use **rmvpe** for f0.
- If you hear nothing, check the M-TRACK's physical input-gain knob and that the mic is in the analog combo jack.
- `faiss` logs a harmless `swigfaiss_avx2` warning and falls back to the non-AVX2 build.
