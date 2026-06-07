# zeta.md — Windows + GeForce GTX 1660 realtime RVC setup

Setup notes for running this repo's **realtime voice-changer GUI** (`gui_v1.py`) on
**Windows 10/11** with an **NVIDIA GeForce GTX 1660 / 1660 Super / 1660 Ti** (6 GB),
using a Python **virtual environment** + `pip install -r` and a CUDA build of PyTorch.

The committed Linux setup (`README.md`, `run-realtime-gui.sh`, `.markscomp.env`) does
**not** apply on Windows — PipeWire/`pactl` routing is replaced by VB-CABLE here.

---

## 0. What's special about the GTX 1660

- **Architecture:** Turing, compute capability **sm_75**, **6 GB** VRAM. Plenty for
  realtime inference, but slower than high-end cards — expect higher latency.
- **fp16 is auto-disabled.** `configs/config.py` forces `is_half = False` for any GPU
  whose name contains `"16"` (the 16-series). This is deliberate: fp16 on these cards
  produces NaNs / silence. You do **not** set any flag — the code detects the 1660 and
  runs **fp32** automatically. It also picks the lower-VRAM `x_pad/x_query/x_center/x_max`
  profile, which fits 6 GB comfortably.
- Net effect: you get correct audio out of the box, just don't expect 4090-class latency.

---

## 1. Prerequisites

1. **NVIDIA driver** ≥ 452.39 (anything current is fine) — required for CUDA 11.8 wheels.
   Verify the card is visible:
   ```bat
   nvidia-smi
   ```
2. **Python 3.10** — hard requirement. The realtime stack depends on **fairseq 0.12.2**,
   which only works on Python 3.8–3.10 (3.11+ breaks on fairseq/omegaconf dataclass
   defaults). Install "Python 3.10.x" from python.org and tick **"Add python.exe to PATH"**.
   Confirm the launcher sees it:
   ```bat
   py -3.10 --version
   ```
3. **Microsoft C++ Build Tools** (only if fairseq has no prebuilt wheel for your setup and
   pip tries to compile it). Install "Build Tools for Visual Studio" → "Desktop development
   with C++". Most of the time the fairseq wheel installs without this.
4. **Git** (to clone) and this repo checked out locally.

---

## 2. Create and activate the virtual environment

From the repo root, in **PowerShell** or **cmd**:

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip wheel "setuptools<81"
```

Your prompt should now show `(.venv)`. Everything below runs inside it.

> If PowerShell blocks activation with an execution-policy error, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (or use `cmd` instead).

---

## 3. Install PyTorch with the right CUDA (do this BEFORE requirements)

The 1660 wants the **CUDA 11.8** build of torch (matches the comment in
`requirements-win-for-realtime_vc_gui.txt` and is well-supported on Turing):

```bat
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

(CUDA 12.1 — `--index-url https://download.pytorch.org/whl/cu121` — also works on the 1660
if you prefer newer; 11.8 is the safest default.)

Verify CUDA is live and the card is detected as fp32:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see `True` and `NVIDIA GeForce GTX 1660 ...`.

---

## 4. Install the realtime-GUI requirements

The repo ships `requirements-win-for-realtime_vc_gui.txt`, but it has **two known traps**
(see `README.md`): it lists `wave` (a bogus PyPI package that drags in `MySQL-python` and
aborts the whole install — Python already has `wave` built in) and `PySimpleGUI` (the code
actually imports the open fork **`FreeSimpleGUI`**).

Easiest path — create a local `requirements-zeta.txt` with those two fixed:

```text
# requirements-zeta.txt  (torch/torchaudio already installed in step 3)
einops
fairseq
flask
flask_cors
gin
gin_config
librosa
local_attention
matplotlib
praat-parselmouth
pyworld
PyYAML
resampy
scikit_learn
scipy
SoundFile
tensorboard
tqdm
FreeSimpleGUI
sounddevice<0.5.0
faiss-cpu
gradio
noisereduce
torchfcpe
torchcrepe
python-dotenv
```

Then:

```bat
pip install -r requirements-zeta.txt
```

> Alternatively use the shipped file but skip the traps:
> open `requirements-win-for-realtime_vc_gui.txt`, delete the `wave` line, and change
> `PySimpleGUI` → `FreeSimpleGUI`, then `pip install -r requirements-win-for-realtime_vc_gui.txt`.

### fairseq notes
- If `pip install fairseq` fails to build, install the C++ Build Tools (step 1.3) and retry,
  or install from a git checkout: `pip install git+https://github.com/facebookresearch/fairseq.git`.
- The committed patch in `infer/lib/rtrvc.py` already registers
  `fairseq.data.dictionary.Dictionary` as a torch "safe global", so HuBERT loads under
  PyTorch ≥ 2.6's `weights_only` default. No action needed.

---

## 5. Download the models

Run the Windows downloader (fetches HuBERT, RMVPE, and the pretrained assets):

```bat
tools\dlmodels.bat
```

If it fails (proxy/network), grab `assets/hubert/hubert_base.pt` and
`assets/rmvpe/rmvpe.pt` manually from the upstream release and drop them in those folders.

---

## 6. Audio routing into Discord (Windows = VB-CABLE, not PipeWire)

Windows has no `pactl`/null-sink, so use a virtual audio cable:

1. Install **VB-Audio Virtual Cable** (VB-CABLE) — free. After install you get two devices:
   **"CABLE Input"** (a playback device) and **"CABLE Output"** (a recording device).
2. In the **RVC GUI**:
   - **Sound API:** `MME` (most reliable; choose the *same* API for input and output, per
     the upstream changelog note).
   - **Input Device:** your real microphone.
   - **Output Device:** **`CABLE Input (VB-Audio Virtual Cable)`**.
3. In **Discord** → Settings → Voice & Video:
   - **Input Device:** **`CABLE Output (VB-Audio Virtual Cable)`**.
   - Turn **off** Discord's Noise Suppression / Echo Cancellation / AGC (RVC already
     conditions the audio).

Chain:
```
mic → RVC GUI [input: mic] → GTX 1660 voice conversion (fp32)
    → RVC GUI [output: CABLE Input] → CABLE Output → Discord input
```

---

## 7. Launch

With the venv active:

```bat
python gui_v1.py
```

(`go-realtime-gui.bat` also works, but it calls a bundled `runtime\python.exe`; since
you're using `.venv`, prefer `python gui_v1.py` from the activated environment.)

In the GUI: load your **model `.pth`** + **`.index`**, set devices as in step 6, then click
**Start Audio Conversion**.

### 7.1 Create a desktop shortcut (one-click launch)

So you don't have to open a terminal and activate the venv every time, make a small launcher
`.bat` and drop a shortcut to it on the desktop.

1. In the repo root, create **`launch-zeta.bat`** with this content (adjust the `cd /d` path
   if your checkout lives elsewhere):
   ```bat
   @echo off
   cd /d "%~dp0"
   call .venv\Scripts\activate.bat
   python gui_v1.py
   ```
   `%~dp0` makes the script `cd` to its own folder, so the shortcut works no matter where
   Windows launches it from. Save it next to `gui_v1.py`.

2. Make the desktop shortcut. Either:
   - **GUI way:** right-click `launch-zeta.bat` → **Show more options** → **Send to** →
     **Desktop (create shortcut)**. Rename the new desktop shortcut to e.g. **"RVC Voice"**.
   - **PowerShell way** (one command, creates it directly on your desktop):
     ```powershell
     $s=(New-Object -ComObject WScript.Shell).CreateShortcut("$env:USERPROFILE\Desktop\RVC Voice.lnk"); $s.TargetPath="$PWD\launch-zeta.bat"; $s.WorkingDirectory="$PWD"; $s.Save()
     ```

3. *(Optional polish)* Right-click the shortcut → **Properties**:
   - **Run:** `Minimized` — hides the console window once the GUI is up.
   - **Change Icon…** — point it at any `.ico` if you want a custom icon (the `.bat` has none
     by default).

Double-clicking the shortcut now activates the venv and opens the RVC GUI directly.

---

## 8. Tuning for the 1660 (fp32, 6 GB)

- **f0 method:** `rmvpe` for best quality; `fcpe` is lighter if you need more headroom.
- **Block time:** start around **0.25–0.40 s**. The 1660 in fp32 won't sustain the
  ~0.10–0.15 s a 4090 can; if you hear stutters/dropouts, raise block time.
- **Crossfade / extra inference time:** keep modest; raising them adds latency.
- **Index rate:** 0.3–0.75 is usually fine; higher uses more CPU for the faiss search.
- **n_cpu:** only matters for the `harvest` f0 detector; ignored by rmvpe/fcpe/crepe.
- If you hit CUDA out-of-memory (unlikely at 6 GB fp32 for inference), lower block time and
  close other GPU apps; the config already selects the small-VRAM padding profile.

---

## 9. Quick troubleshooting

| Symptom | Fix |
|---|---|
| `pip` aborts installing `wave` / pulls `MySQL-python` | Remove `wave` from the requirements (step 4). |
| `ModuleNotFoundError: PySimpleGUI` | Install `FreeSimpleGUI` (step 4). |
| `torch.cuda.is_available()` is `False` | Reinstall the **cu118** torch wheel (step 3); update NVIDIA driver. |
| Output is silent / NaN | You forced fp16 somewhere — let `config.py` auto-pick fp32 for the 1660; don't override. |
| Discord doesn't list the cable | Reinstall VB-CABLE, reboot, and reopen Discord. |
| fairseq build error | Install MS C++ Build Tools, or `pip install git+https://github.com/facebookresearch/fairseq.git`. |
| Python is 3.11/3.12 | Recreate the venv with `py -3.10` — fairseq won't work on 3.11+. |
