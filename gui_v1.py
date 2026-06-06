import os
import sys
from dotenv import load_dotenv
import shutil
import subprocess

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import multiprocessing

flag_vc = False


def native_file_picker(title, initial_dir, pattern_name, pattern):
    """Open the native GTK file chooser (zenity) and return the chosen path.

    Falls back to an empty string on cancel or if zenity is unavailable
    (the GUI then keeps whatever is already typed in the field)."""
    cmd = ["zenity", "--file-selection", "--title", title]
    if initial_dir and os.path.isdir(initial_dir):
        # trailing sep tells zenity to open *inside* the folder
        cmd += ["--filename", os.path.join(initial_dir, "")]
    if pattern:
        cmd += [
            "--file-filter", "%s | %s" % (pattern_name, pattern),
            "--file-filter", "All files | *",
        ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result


class Harvest(multiprocessing.Process):
    def __init__(self, inp_q, opt_q):
        multiprocessing.Process.__init__(self)
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self):
        import numpy as np
        import pyworld

        while 1:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)


if __name__ == "__main__":
    import json
    import multiprocessing
    import re
    import threading
    import time
    import traceback
    from multiprocessing import Queue, cpu_count
    from queue import Empty

    import librosa
    from tools.torchgate import TorchGate
    import numpy as np
    import FreeSimpleGUI as sg
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat

    from infer.lib import rtrvc as rvc_for_realtime
    from infer.lib.rt_debug import get_logger
    from infer.lib.rt_eq import StreamEQ
    from infer.lib import rt_priority
    from infer.lib.rt_waterfall import Waterfall, blank_png
    from i18n.i18n import I18nAuto
    from configs.config import Config

    i18n = I18nAuto(language="en_US")
    dbg = get_logger()

    # Spectrogram waterfall display size (pixels).
    WF_W, WF_H = 430, 170
    WF_BLANK = blank_png(WF_W, WF_H)

    # device = rvc_for_realtime.config.device
    # device = torch.device(
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else ("mps" if torch.backends.mps.is_available() else "cpu")
    # )
    current_dir = os.getcwd()
    inp_q = Queue()
    opt_q = Queue()
    n_cpu = min(cpu_count(), 8)
    for _ in range(n_cpu):
        p = Harvest(inp_q, opt_q)
        p.daemon = True
        p.start()

    # Real-time CPU scheduling. Pin the main process to the performance cores
    # (threads created later -- audio callback, torch -- inherit this), so the
    # hybrid-CPU scheduler can't park latency-critical work on E-cores. The
    # audio thread itself is given SCHED_FIFO from inside the first callback.
    RT_PRIO = int(os.environ.get("RVC_RT_PRIO", "20"))  # 0 disables the FIFO boost
    if os.environ.get("RVC_CPU_AFFINITY", "1") not in ("0", "false", "off"):
        _pcores = rt_priority.pin_to_pcores()
        dbg.note(
            "CPU affinity -> P-cores %s" % _pcores
            if _pcores
            else "CPU affinity: P-cores not detected (left unpinned)"
        )

    class GUIConfig:
        def __init__(self) -> None:
            self.pth_path: str = ""
            self.index_path: str = ""
            self.pitch: int = 0
            self.formant=0.0
            self.sr_type: str = "sr_model"
            self.block_time: float = 0.25  # s
            self.threhold: int = -60
            self.crossfade_time: float = 0.05
            self.extra_time: float = 2.5
            self.I_noise_reduce: bool = False
            self.O_noise_reduce: bool = False
            self.use_pv: bool = False
            self.use_half: bool = True
            self.nr_strength: float = 0.9
            self.sola_search_ms: float = 10.0
            self.eq_in_enable: bool = False
            self.eq_in = [0.0, 0.0, 0.0]
            self.eq_out_enable: bool = False
            self.eq_out = [0.0, 0.0, 0.0]
            self.rms_mix_rate: float = 0.0
            self.index_rate: float = 0.0
            self.n_cpu: int = min(n_cpu, 4)
            self.f0method: str = "fcpe"
            self.sg_hostapi: str = ""
            self.wasapi_exclusive: bool = False
            self.sg_input_device: str = ""
            self.sg_output_device: str = ""

    class GUI:
        def __init__(self) -> None:
            self.gui_config = GUIConfig()
            self.config = Config()
            self.function = "vc"
            self.delay_time = 0
            self.hostapis = None
            self.input_devices = None
            self.output_devices = None
            self.input_devices_indices = None
            self.output_devices_indices = None
            self.stream = None
            self.update_devices()
            self.launcher()

        def load(self):
            try:
                if not os.path.exists("configs/inuse/config.json"):
                    shutil.copy("configs/config.json", "configs/inuse/config.json")
                with open("configs/inuse/config.json", "r") as j:
                    data = json.load(j)
                    data["sr_model"] = data["sr_type"] == "sr_model"
                    data["sr_device"] = data["sr_type"] == "sr_device"
                    data["pm"] = data["f0method"] == "pm"
                    data["harvest"] = data["f0method"] == "harvest"
                    data["crepe"] = data["f0method"] == "crepe"
                    data["rmvpe"] = data["f0method"] == "rmvpe"
                    data["fcpe"] = data["f0method"] == "fcpe"
                    if data["sg_hostapi"] in self.hostapis:
                        self.update_devices(hostapi_name=data["sg_hostapi"])
                        if (
                            data["sg_input_device"] not in self.input_devices
                            or data["sg_output_device"] not in self.output_devices
                        ):
                            self.update_devices()
                            data["sg_hostapi"] = self.hostapis[0]
                            data["sg_input_device"] = self.input_devices[
                                self.input_devices_indices.index(sd.default.device[0])
                            ]
                            data["sg_output_device"] = self.output_devices[
                                self.output_devices_indices.index(sd.default.device[1])
                            ]
                    else:
                        data["sg_hostapi"] = self.hostapis[0]
                        data["sg_input_device"] = self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ]
                        data["sg_output_device"] = self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ]
            except:
                with open("configs/inuse/config.json", "w") as j:
                    data = {
                        "pth_path": "",
                        "index_path": "",
                        "sg_hostapi": self.hostapis[0],
                        "sg_wasapi_exclusive": False,
                        "sg_input_device": self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ],
                        "sg_output_device": self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ],
                        "sr_type": "sr_model",
                        "threhold": -60,
                        "pitch": 0,
                        "formant": 0.0,
                        "index_rate": 0,
                        "rms_mix_rate": 0,
                        "block_time": 0.25,
                        "crossfade_length": 0.05,
                        "extra_time": 2.5,
                        "n_cpu": 4,
                        "f0method": "rmvpe",
                        "use_jit": False,
                        "use_pv": False,
                        "use_half": True,
                        "nr_strength": 0.9,
                        "sola_search_ms": 10,
                        "eq_in_enable": False,
                        "eq_in_low": 0,
                        "eq_in_mid": 0,
                        "eq_in_high": 0,
                        "eq_out_enable": False,
                        "eq_out_low": 0,
                        "eq_out_mid": 0,
                        "eq_out_high": 0,
                    }
                    data["sr_model"] = data["sr_type"] == "sr_model"
                    data["sr_device"] = data["sr_type"] == "sr_device"
                    data["pm"] = data["f0method"] == "pm"
                    data["harvest"] = data["f0method"] == "harvest"
                    data["crepe"] = data["f0method"] == "crepe"
                    data["rmvpe"] = data["f0method"] == "rmvpe"
                    data["fcpe"] = data["f0method"] == "fcpe"
            return data

        def launcher(self):
            data = self.load()
            self.config.use_jit = False  # data.get("use_jit", self.config.use_jit)
            # Default folders for the model picker: prefer the curated
            # ../voices/{path,index} layout, else fall back to the repo defaults.
            voices_dir = os.path.abspath(os.path.join(now_dir, "..", "voices"))
            self.pth_init_dir = os.path.join(voices_dir, "path")
            self.index_init_dir = os.path.join(voices_dir, "index")
            if not os.path.isdir(self.pth_init_dir):
                self.pth_init_dir = os.path.join(now_dir, "assets", "weights")
            if not os.path.isdir(self.index_init_dir):
                self.index_init_dir = os.path.join(now_dir, "logs")
            sg.theme("LightBlue3")

            def cap(text):
                """One-line grey explanation row placed under a control."""
                return [
                    sg.Text(
                        text,
                        font=("Any", 10),
                        text_color="#444444",
                        pad=((22, 0), (0, 6)),
                    )
                ]

            layout = [
                [
                    sg.Frame(
                        title=i18n("加载模型"),
                        layout=[
                            [
                                sg.Input(
                                    default_text=data.get("pth_path", ""),
                                    key="pth_path",
                                    size=(58, 1),
                                ),
                                sg.Button(
                                    i18n("选择.pth文件"),
                                    key="browse_pth",
                                ),
                            ],
                            [
                                sg.Input(
                                    default_text=data.get("index_path", ""),
                                    key="index_path",
                                    size=(58, 1),
                                ),
                                sg.Button(
                                    i18n("选择.index文件"),
                                    key="browse_index",
                                ),
                            ],
                        ],
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("设备类型")),
                                sg.Combo(
                                    self.hostapis,
                                    key="sg_hostapi",
                                    default_value=data.get("sg_hostapi", ""),
                                    enable_events=True,
                                    size=(20, 1),
                                ),
                                sg.Checkbox(
                                    i18n("独占 WASAPI 设备"),
                                    key="sg_wasapi_exclusive",
                                    default=data.get("sg_wasapi_exclusive", False),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("输入设备")),
                                sg.Combo(
                                    self.input_devices,
                                    key="sg_input_device",
                                    default_value=data.get("sg_input_device", ""),
                                    enable_events=True,
                                    size=(45, 1),
                                ),
                            ],
                            [
                                sg.Text(i18n("输出设备")),
                                sg.Combo(
                                    self.output_devices,
                                    key="sg_output_device",
                                    default_value=data.get("sg_output_device", ""),
                                    enable_events=True,
                                    size=(45, 1),
                                ),
                            ],
                            [
                                sg.Button(i18n("重载设备列表"), key="reload_devices"),
                                sg.Radio(
                                    i18n("使用模型采样率"),
                                    "sr_type",
                                    key="sr_model",
                                    default=data.get("sr_model", True),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    i18n("使用设备采样率"),
                                    "sr_type",
                                    key="sr_device",
                                    default=data.get("sr_device", False),
                                    enable_events=True,
                                ),
                                sg.Text(i18n("采样率:")),
                                sg.Text("", key="sr_stream"),
                            ],
                        ],
                        title=i18n("音频设备"),
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("响应阈值")),
                                sg.Slider(
                                    range=(-60, 0),
                                    key="threhold",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("threhold", -60),
                                    enable_events=True,
                                ),
                            ],
                            cap("Noise gate. Mic audio quieter than this (dB) is muted. -60 = gate OFF; raise toward -45 to silence room tone between words."),
                            [
                                sg.Text(i18n("音调设置")),
                                sg.Slider(
                                    range=(-16, 16),
                                    key="pitch",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("pitch", 0),
                                    enable_events=True,
                                ),
                            ],
                            cap("Pitch shift in semitones (+12 = up one octave). Match your voice to the model; large shifts can sound less natural."),
                            [
                                sg.Text(i18n("性别因子/声线粗细")),
                                sg.Slider(
                                    range=(-2, 2),
                                    key="formant",
                                    resolution=0.05,
                                    orientation="h",
                                    default_value=data.get("formant", 0.0),
                                    enable_events=True,
                                ),
                            ],
                            cap("Voice timbre / vocal-tract size, independent of pitch. Negative = deeper/larger, positive = brighter/smaller."),
                            [
                                sg.Text(i18n("Index Rate")),
                                sg.Slider(
                                    range=(0.0, 1.0),
                                    key="index_rate",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("index_rate", 0),
                                    enable_events=True,
                                ),
                            ],
                            cap("How hard to snap features onto the trained voice (0-1). Higher = closer timbre but less continuous; lower = smoother, more of your own voice."),
                            [
                                sg.Text(i18n("响度因子")),
                                sg.Slider(
                                    range=(0.0, 1.0),
                                    key="rms_mix_rate",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("rms_mix_rate", 0),
                                    enable_events=True,
                                ),
                            ],
                            cap("Loudness follow. 0 = use the model's own loudness; 1 = track your mic's volume envelope. Mid values can pump on quiet passages."),
                            [
                                sg.Text(i18n("音高算法")),
                                sg.Radio(
                                    "pm",
                                    "f0method",
                                    key="pm",
                                    default=data.get("pm", False),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "harvest",
                                    "f0method",
                                    key="harvest",
                                    default=data.get("harvest", False),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "crepe",
                                    "f0method",
                                    key="crepe",
                                    default=data.get("crepe", False),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "rmvpe",
                                    "f0method",
                                    key="rmvpe",
                                    default=data.get("rmvpe", False),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "fcpe",
                                    "f0method",
                                    key="fcpe",
                                    default=data.get("fcpe", True),
                                    enable_events=True,
                                ),
                            ],
                            cap("Pitch detector. rmvpe = most accurate/robust (recommended); fcpe = fastest; crepe = accurate but heavier; pm/harvest = legacy."),
                        ],
                        title=i18n("常规设置"),
                    ),
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("采样长度")),
                                sg.Slider(
                                    range=(0.02, 1.5),
                                    key="block_time",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("block_time", 0.25),
                                    enable_events=True,
                                ),
                            ],
                            cap("Chunk processed per step (s). Smaller = lower latency but a tighter per-block deadline; larger = safer but more delay. Applied on Start."),
                            [
                                sg.Text(i18n("harvest进程数")),
                                sg.Slider(
                                    range=(1, n_cpu),
                                    key="n_cpu",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get(
                                        "n_cpu", min(self.gui_config.n_cpu, n_cpu)
                                    ),
                                    enable_events=True,
                                ),
                            ],
                            cap("CPU worker processes for the 'harvest' pitch detector only. Ignored by rmvpe/fcpe/crepe."),
                            [
                                sg.Text(i18n("淡入淡出长度")),
                                sg.Slider(
                                    range=(0.01, 0.15),
                                    key="crossfade_length",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("crossfade_length", 0.05),
                                    enable_events=True,
                                ),
                            ],
                            cap("Crossfade between consecutive output chunks (s). Longer hides splices but adds delay and can blur transients. Applied on Start."),
                            [
                                sg.Text(i18n("额外推理时长")),
                                sg.Slider(
                                    range=(0.05, 5.00),
                                    key="extra_time",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("extra_time", 2.5),
                                    enable_events=True,
                                ),
                            ],
                            cap("Extra past audio fed as context each step (s). More = steadier conversion but higher latency. 1-2.5s typical; 5s is heavy. Applied on Start."),
                            [
                                sg.Checkbox(
                                    i18n("输入降噪"),
                                    key="I_noise_reduce",
                                    enable_events=True,
                                ),
                                sg.Checkbox(
                                    i18n("输出降噪"),
                                    key="O_noise_reduce",
                                    enable_events=True,
                                ),
                                sg.Checkbox(
                                    i18n("启用相位声码器"),
                                    key="use_pv",
                                    default=data.get("use_pv", False),
                                    enable_events=True,
                                ),
                            ],
                            cap("Input denoise = clean the mic before conversion. Output denoise = clean the converted voice. Phase vocoder = smooths chunk splices (fixes choppiness)."),
                        ],
                        title=i18n("性能设置"),
                    ),
                ],
                [
                    sg.Frame(
                        title="Input EQ (shapes the mic before conversion)",
                        layout=[
                            [
                                sg.Checkbox(
                                    "Enable input EQ",
                                    key="eq_in_enable",
                                    default=data.get("eq_in_enable", False),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Low ", size=(4, 1)),
                                sg.Slider(
                                    range=(-12, 12),
                                    key="eq_in_low",
                                    resolution=0.5,
                                    orientation="h",
                                    default_value=data.get("eq_in_low", 0),
                                    enable_events=True,
                                ),
                            ],
                            cap("Low shelf ~120 Hz: warmth/rumble (dB). Cut to tame proximity boom on the AT4040."),
                            [
                                sg.Text("Mid ", size=(4, 1)),
                                sg.Slider(
                                    range=(-12, 12),
                                    key="eq_in_mid",
                                    resolution=0.5,
                                    orientation="h",
                                    default_value=data.get("eq_in_mid", 0),
                                    enable_events=True,
                                ),
                            ],
                            cap("Peak ~1.5 kHz: presence/nasal (dB)."),
                            [
                                sg.Text("High", size=(4, 1)),
                                sg.Slider(
                                    range=(-12, 12),
                                    key="eq_in_high",
                                    resolution=0.5,
                                    orientation="h",
                                    default_value=data.get("eq_in_high", 0),
                                    enable_events=True,
                                ),
                            ],
                            cap("High shelf ~6 kHz: air/sibilance (dB). Cut to reduce hiss feeding the model."),
                        ],
                    ),
                    sg.Frame(
                        title="Output EQ (shapes the converted voice)",
                        layout=[
                            [
                                sg.Checkbox(
                                    "Enable output EQ",
                                    key="eq_out_enable",
                                    default=data.get("eq_out_enable", False),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text("Low ", size=(4, 1)),
                                sg.Slider(
                                    range=(-12, 12),
                                    key="eq_out_low",
                                    resolution=0.5,
                                    orientation="h",
                                    default_value=data.get("eq_out_low", 0),
                                    enable_events=True,
                                ),
                            ],
                            cap("Low shelf ~120 Hz: body/weight of the output voice (dB)."),
                            [
                                sg.Text("Mid ", size=(4, 1)),
                                sg.Slider(
                                    range=(-12, 12),
                                    key="eq_out_mid",
                                    resolution=0.5,
                                    orientation="h",
                                    default_value=data.get("eq_out_mid", 0),
                                    enable_events=True,
                                ),
                            ],
                            cap("Peak ~1.5 kHz: intelligibility/presence (dB)."),
                            [
                                sg.Text("High", size=(4, 1)),
                                sg.Slider(
                                    range=(-12, 12),
                                    key="eq_out_high",
                                    resolution=0.5,
                                    orientation="h",
                                    default_value=data.get("eq_out_high", 0),
                                    enable_events=True,
                                ),
                            ],
                            cap("High shelf ~6 kHz: air/brightness (dB). Boost for sparkle, cut to de-ess."),
                        ],
                    ),
                ],
                [
                    sg.Frame(
                        title="Quality (advanced) — applied on Start",
                        layout=[
                            [
                                sg.Checkbox(
                                    "Half precision (FP16) — uncheck for FP32 (cleaner, slower)",
                                    key="use_half",
                                    default=data.get("use_half", True),
                                ),
                            ],
                            [
                                sg.Text("Noise-reduce strength"),
                                sg.Slider(
                                    range=(0.0, 1.0),
                                    key="nr_strength",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("nr_strength", 0.9),
                                ),
                                sg.Text("(applies to In/Out denoise)"),
                            ],
                            [
                                sg.Text("SOLA search window (ms)"),
                                sg.Slider(
                                    range=(1, 40),
                                    key="sola_search_ms",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("sola_search_ms", 10),
                                ),
                                sg.Text("(wider = more robust splice alignment)"),
                            ],
                        ],
                    ),
                    sg.Frame(
                        title="Spectrogram waterfall (P-core driven) — time →, low Hz at bottom",
                        layout=[
                            [sg.Text("Input  (mic, post-gate / post-EQ)")],
                            [sg.Image(data=WF_BLANK, key="wf_in")],
                            [sg.Text("Output (converted voice, post-EQ)")],
                            [sg.Image(data=WF_BLANK, key="wf_out")],
                        ],
                    ),
                ],
                [
                    sg.Button(i18n("开始音频转换"), key="start_vc"),
                    sg.Button(i18n("停止音频转换"), key="stop_vc"),
                    sg.Radio(
                        i18n("输入监听"),
                        "function",
                        key="im",
                        default=False,
                        enable_events=True,
                    ),
                    sg.Radio(
                        i18n("输出变声"),
                        "function",
                        key="vc",
                        default=True,
                        enable_events=True,
                    ),
                    sg.Text(i18n("算法延迟(ms):")),
                    sg.Text("0", key="delay_time"),
                    sg.Text(i18n("推理时间(ms):")),
                    sg.Text("0", key="infer_time"),
                ],
            ]
            self.window = sg.Window("RVC - GUI", layout=layout, finalize=True)
            self.event_handler()

        def event_handler(self):
            global flag_vc
            while True:
                # Timeout drives the ~20fps scope redraw between real events.
                event, values = self.window.read(timeout=50)
                if event == sg.WINDOW_CLOSED:
                    self.stop_stream()
                    exit()
                self._update_waterfall()
                if event in (sg.TIMEOUT_KEY, None):
                    continue
                if event == "browse_pth":
                    chosen = native_file_picker(
                        i18n("选择.pth文件"),
                        self.pth_init_dir,
                        "RVC model (*.pth)",
                        "*.pth",
                    )
                    if chosen:
                        self.window["pth_path"].update(value=chosen)
                    continue
                if event == "browse_index":
                    chosen = native_file_picker(
                        i18n("选择.index文件"),
                        self.index_init_dir,
                        "RVC index (*.index)",
                        "*.index",
                    )
                    if chosen:
                        self.window["index_path"].update(value=chosen)
                    continue
                if event == "reload_devices" or event == "sg_hostapi":
                    self.gui_config.sg_hostapi = values["sg_hostapi"]
                    self.update_devices(hostapi_name=values["sg_hostapi"])
                    if self.gui_config.sg_hostapi not in self.hostapis:
                        self.gui_config.sg_hostapi = self.hostapis[0]
                    self.window["sg_hostapi"].Update(values=self.hostapis)
                    self.window["sg_hostapi"].Update(value=self.gui_config.sg_hostapi)
                    if (
                        self.gui_config.sg_input_device not in self.input_devices
                        and len(self.input_devices) > 0
                    ):
                        self.gui_config.sg_input_device = self.input_devices[0]
                    self.window["sg_input_device"].Update(values=self.input_devices)
                    self.window["sg_input_device"].Update(
                        value=self.gui_config.sg_input_device
                    )
                    if (
                        self.gui_config.sg_output_device not in self.output_devices
                        and len(self.output_devices) > 0
                    ):
                        self.gui_config.sg_output_device = self.output_devices[0]
                    self.window["sg_output_device"].Update(values=self.output_devices)
                    self.window["sg_output_device"].Update(
                        value=self.gui_config.sg_output_device
                    )
                if event == "start_vc" and not flag_vc:
                    if self.set_values(values) == True:
                        printt("cuda_is_available: %s", torch.cuda.is_available())
                        self.start_vc()
                        settings = {
                            "pth_path": values["pth_path"],
                            "index_path": values["index_path"],
                            "sg_hostapi": values["sg_hostapi"],
                            "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],
                            "sg_input_device": values["sg_input_device"],
                            "sg_output_device": values["sg_output_device"],
                            "sr_type": ["sr_model", "sr_device"][
                                [
                                    values["sr_model"],
                                    values["sr_device"],
                                ].index(True)
                            ],
                            "threhold": values["threhold"],
                            "pitch": values["pitch"],
                            "rms_mix_rate": values["rms_mix_rate"],
                            "index_rate": values["index_rate"],
                            # "device_latency": values["device_latency"],
                            "block_time": values["block_time"],
                            "crossfade_length": values["crossfade_length"],
                            "extra_time": values["extra_time"],
                            "n_cpu": values["n_cpu"],
                            # "use_jit": values["use_jit"],
                            "use_jit": False,
                            "use_pv": values["use_pv"],
                            "use_half": values["use_half"],
                            "nr_strength": values["nr_strength"],
                            "sola_search_ms": values["sola_search_ms"],
                            "eq_in_enable": values["eq_in_enable"],
                            "eq_in_low": values["eq_in_low"],
                            "eq_in_mid": values["eq_in_mid"],
                            "eq_in_high": values["eq_in_high"],
                            "eq_out_enable": values["eq_out_enable"],
                            "eq_out_low": values["eq_out_low"],
                            "eq_out_mid": values["eq_out_mid"],
                            "eq_out_high": values["eq_out_high"],
                            "f0method": ["pm", "harvest", "crepe", "rmvpe", "fcpe"][
                                [
                                    values["pm"],
                                    values["harvest"],
                                    values["crepe"],
                                    values["rmvpe"],
                                    values["fcpe"],
                                ].index(True)
                            ],
                        }
                        with open("configs/inuse/config.json", "w") as j:
                            json.dump(settings, j)
                        if self.stream is not None:
                            self.delay_time = (
                                self.stream.latency[-1]
                                + values["block_time"]
                                + values["crossfade_length"]
                                + 0.01
                            )
                        if values["I_noise_reduce"]:
                            self.delay_time += min(values["crossfade_length"], 0.04)
                        self.window["sr_stream"].update(self.gui_config.samplerate)
                        self.window["delay_time"].update(
                            int(np.round(self.delay_time * 1000))
                        )
                # Parameter hot update
                if event == "threhold":
                    self.gui_config.threhold = values["threhold"]
                elif event == "pitch":
                    self.gui_config.pitch = values["pitch"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_key(values["pitch"])
                elif event == "formant":
                    self.gui_config.formant = values["formant"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_formant(values["formant"])
                elif event == "index_rate":
                    self.gui_config.index_rate = values["index_rate"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_index_rate(values["index_rate"])
                elif event == "rms_mix_rate":
                    self.gui_config.rms_mix_rate = values["rms_mix_rate"]
                elif event in ["pm", "harvest", "crepe", "rmvpe", "fcpe"]:
                    self.gui_config.f0method = event
                elif event == "I_noise_reduce":
                    self.gui_config.I_noise_reduce = values["I_noise_reduce"]
                    if self.stream is not None:
                        self.delay_time += (
                            1 if values["I_noise_reduce"] else -1
                        ) * min(values["crossfade_length"], 0.04)
                        self.window["delay_time"].update(
                            int(np.round(self.delay_time * 1000))
                        )
                elif event == "O_noise_reduce":
                    self.gui_config.O_noise_reduce = values["O_noise_reduce"]
                elif event == "use_pv":
                    self.gui_config.use_pv = values["use_pv"]
                elif event in ["eq_in_enable", "eq_in_low", "eq_in_mid", "eq_in_high"]:
                    self.gui_config.eq_in_enable = values["eq_in_enable"]
                    self.gui_config.eq_in = [
                        values["eq_in_low"],
                        values["eq_in_mid"],
                        values["eq_in_high"],
                    ]
                    if hasattr(self, "eq_in"):
                        self.eq_in.set_gains(
                            self.gui_config.eq_in_enable, self.gui_config.eq_in
                        )
                elif event in [
                    "eq_out_enable",
                    "eq_out_low",
                    "eq_out_mid",
                    "eq_out_high",
                ]:
                    self.gui_config.eq_out_enable = values["eq_out_enable"]
                    self.gui_config.eq_out = [
                        values["eq_out_low"],
                        values["eq_out_mid"],
                        values["eq_out_high"],
                    ]
                    if hasattr(self, "eq_out"):
                        self.eq_out.set_gains(
                            self.gui_config.eq_out_enable, self.gui_config.eq_out
                        )
                elif event in ["vc", "im"]:
                    self.function = event
                elif event == "stop_vc" or event != "start_vc":
                    # Other parameters do not support hot update
                    self.stop_stream()

        def _update_waterfall(self):
            """Advance and redraw the spectrogram waterfalls when a new audio
            block has arrived. Runs on the (P-core-pinned) GUI thread."""
            if not hasattr(self, "_wf_seq") or self._wf_seq == self._wf_drawn_seq:
                return  # no new block since last draw -> nothing to scroll
            self._wf_drawn_seq = self._wf_seq
            for wf, block, key in (
                (self._wf_in, self._wf_in_block, "wf_in"),
                (self._wf_out, self._wf_out_block, "wf_out"),
            ):
                if block is None:
                    continue
                wf.push(block)
                self.window[key].update(data=wf.render_png())

        def set_values(self, values):
            if len(values["pth_path"].strip()) == 0:
                sg.popup(i18n("请选择pth文件"))
                return False
            if len(values["index_path"].strip()) == 0:
                sg.popup(i18n("请选择index文件"))
                return False
            pattern = re.compile("[^\x00-\x7F]+")
            if pattern.findall(values["pth_path"]):
                sg.popup(i18n("pth文件路径不可包含中文"))
                return False
            if pattern.findall(values["index_path"]):
                sg.popup(i18n("index文件路径不可包含中文"))
                return False
            self.set_devices(values["sg_input_device"], values["sg_output_device"])
            self.config.use_jit = False  # values["use_jit"]
            # self.device_latency = values["device_latency"]
            self.gui_config.sg_hostapi = values["sg_hostapi"]
            self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
            self.gui_config.sg_input_device = values["sg_input_device"]
            self.gui_config.sg_output_device = values["sg_output_device"]
            self.gui_config.pth_path = values["pth_path"]
            self.gui_config.index_path = values["index_path"]
            self.gui_config.sr_type = ["sr_model", "sr_device"][
                [
                    values["sr_model"],
                    values["sr_device"],
                ].index(True)
            ]
            self.gui_config.threhold = values["threhold"]
            self.gui_config.pitch = values["pitch"]
            self.gui_config.formant = values["formant"]
            self.gui_config.block_time = values["block_time"]
            self.gui_config.crossfade_time = values["crossfade_length"]
            self.gui_config.extra_time = values["extra_time"]
            self.gui_config.I_noise_reduce = values["I_noise_reduce"]
            self.gui_config.O_noise_reduce = values["O_noise_reduce"]
            self.gui_config.use_pv = values["use_pv"]
            # Advanced quality knobs (applied on Start)
            self.config.is_half = values["use_half"]
            self.gui_config.nr_strength = values["nr_strength"]
            self.gui_config.sola_search_ms = values["sola_search_ms"]
            self.gui_config.eq_in_enable = values["eq_in_enable"]
            self.gui_config.eq_in = [
                values["eq_in_low"],
                values["eq_in_mid"],
                values["eq_in_high"],
            ]
            self.gui_config.eq_out_enable = values["eq_out_enable"]
            self.gui_config.eq_out = [
                values["eq_out_low"],
                values["eq_out_mid"],
                values["eq_out_high"],
            ]
            self.gui_config.rms_mix_rate = values["rms_mix_rate"]
            self.gui_config.index_rate = values["index_rate"]
            self.gui_config.n_cpu = values["n_cpu"]
            self.gui_config.f0method = ["pm", "harvest", "crepe", "rmvpe", "fcpe"][
                [
                    values["pm"],
                    values["harvest"],
                    values["crepe"],
                    values["rmvpe"],
                    values["fcpe"],
                ].index(True)
            ]
            return True

        def start_vc(self):
            torch.cuda.empty_cache()
            self.rvc = rvc_for_realtime.RVC(
                self.gui_config.pitch,
                self.gui_config.formant,
                self.gui_config.pth_path,
                self.gui_config.index_path,
                self.gui_config.index_rate,
                self.gui_config.n_cpu,
                inp_q,
                opt_q,
                self.config,
                self.rvc if hasattr(self, "rvc") else None,
            )
            self.gui_config.samplerate = (
                self.rvc.tgt_sr
                if self.gui_config.sr_type == "sr_model"
                else self.get_device_samplerate()
            )
            self.gui_config.channels = self.get_device_channels()
            self.zc = self.gui_config.samplerate // 100
            self.block_frame = (
                int(
                    np.round(
                        self.gui_config.block_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.block_frame_16k = 160 * self.block_frame // self.zc
            self.crossfade_frame = (
                int(
                    np.round(
                        self.gui_config.crossfade_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
            # SOLA search window, rounded to whole 10ms (zc) frames; default
            # 10ms reproduces the original behavior (1 * zc).
            self.sola_search_frame = (
                max(
                    1,
                    int(
                        np.round(
                            self.gui_config.sola_search_ms
                            * self.gui_config.samplerate
                            / 1000
                            / self.zc
                        )
                    ),
                )
                * self.zc
            )
            self.extra_frame = (
                int(
                    np.round(
                        self.gui_config.extra_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.input_wav: torch.Tensor = torch.zeros(
                self.extra_frame
                + self.crossfade_frame
                + self.sola_search_frame
                + self.block_frame,
                device=self.config.device,
                dtype=torch.float32,
            )
            self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
            self.input_wav_res: torch.Tensor = torch.zeros(
                160 * self.input_wav.shape[0] // self.zc,
                device=self.config.device,
                dtype=torch.float32,
            )
            self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
            self.sola_buffer: torch.Tensor = torch.zeros(
                self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
            )
            self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
            self.output_buffer: torch.Tensor = self.input_wav.clone()
            self.skip_head = self.extra_frame // self.zc
            self.return_length = (
                self.block_frame + self.sola_buffer_frame + self.sola_search_frame
            ) // self.zc
            self.fade_in_window: torch.Tensor = (
                torch.sin(
                    0.5
                    * np.pi
                    * torch.linspace(
                        0.0,
                        1.0,
                        steps=self.sola_buffer_frame,
                        device=self.config.device,
                        dtype=torch.float32,
                    )
                )
                ** 2
            )
            self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
            self.resampler = tat.Resample(
                orig_freq=self.gui_config.samplerate,
                new_freq=16000,
                dtype=torch.float32,
            ).to(self.config.device)
            if self.rvc.tgt_sr != self.gui_config.samplerate:
                self.resampler2 = tat.Resample(
                    orig_freq=self.rvc.tgt_sr,
                    new_freq=self.gui_config.samplerate,
                    dtype=torch.float32,
                ).to(self.config.device)
            else:
                self.resampler2 = None
            self.tg = TorchGate(
                sr=self.gui_config.samplerate,
                n_fft=4 * self.zc,
                prop_decrease=self.gui_config.nr_strength,
            ).to(self.config.device)
            # Per-stream EQ (state carried across blocks) + scope buffers.
            self.eq_in = StreamEQ(self.gui_config.samplerate)
            self.eq_in.set_gains(self.gui_config.eq_in_enable, self.gui_config.eq_in)
            self.eq_out = StreamEQ(self.gui_config.samplerate)
            self.eq_out.set_gains(self.gui_config.eq_out_enable, self.gui_config.eq_out)
            # Spectrogram waterfalls (FFT/render run on the GUI thread, which is
            # P-core-pinned with the rest of the process). The audio thread only
            # stashes the latest block + bumps a sequence counter.
            self._wf_in = Waterfall(self.gui_config.samplerate, out_w=WF_W, out_h=WF_H)
            self._wf_out = Waterfall(self.gui_config.samplerate, out_w=WF_W, out_h=WF_H)
            self._wf_in_block = None
            self._wf_out_block = None
            self._wf_seq = 0
            self._wf_drawn_seq = -1
            # Per-callback real-time deadline: one block of audio must be
            # produced in less wall-clock time than that block lasts.
            budget_ms = 1000.0 * self.block_frame / self.gui_config.samplerate
            dbg.banner(
                "RVC SESSION START",
                {
                    "device": self.config.device,
                    "is_half": self.config.is_half,
                    "samplerate": self.gui_config.samplerate,
                    "tgt_sr": self.rvc.tgt_sr,
                    "channels": self.gui_config.channels,
                    "block_time_s": self.gui_config.block_time,
                    "block_frame": self.block_frame,
                    "budget_ms": round(budget_ms, 2),
                    "crossfade_s": self.gui_config.crossfade_time,
                    "sola_buffer": self.sola_buffer_frame,
                    "sola_search": self.sola_search_frame,
                    "nr_strength": self.gui_config.nr_strength,
                    "eq_in": "%s %s" % (self.gui_config.eq_in_enable, self.gui_config.eq_in),
                    "eq_out": "%s %s" % (self.gui_config.eq_out_enable, self.gui_config.eq_out),
                    "extra_time_s": self.gui_config.extra_time,
                    "f0method": self.gui_config.f0method,
                    "pitch": self.gui_config.pitch,
                    "formant": self.gui_config.formant,
                    "index_rate": self.gui_config.index_rate,
                    "rms_mix_rate": self.gui_config.rms_mix_rate,
                    "threshold_db": self.gui_config.threhold,
                    "I_noise_reduce": self.gui_config.I_noise_reduce,
                    "O_noise_reduce": self.gui_config.O_noise_reduce,
                    "use_pv": self.gui_config.use_pv,
                    "n_cpu": self.gui_config.n_cpu,
                },
            )
            self.start_stream()

        def start_stream(self):
            global flag_vc
            if not flag_vc:
                flag_vc = True
                if (
                    "WASAPI" in self.gui_config.sg_hostapi
                    and self.gui_config.sg_wasapi_exclusive
                ):
                    extra_settings = sd.WasapiSettings(exclusive=True)
                else:
                    extra_settings = None
                try:
                    self.stream = sd.Stream(
                        callback=self.audio_callback,
                        blocksize=self.block_frame,
                        samplerate=self.gui_config.samplerate,
                        channels=self.gui_config.channels,
                        dtype="float32",
                        extra_settings=extra_settings,
                    )
                    self.stream.start()
                except Exception as e:
                    flag_vc = False
                    self.stream = None
                    printt("Failed to open audio stream: %s", str(e))
                    sg.popup_error(
                        "Could not open the audio device:\n\n%s\n\n"
                        "Set BOTH Input and Output device to 'pulse'. "
                        "Do not pick a raw 'hw:' device (e.g. M-TRACK DUO HD "
                        "(hw:1,0)) — PipeWire already owns it, so PortAudio "
                        "cannot grab it directly." % str(e),
                        title="Audio device unavailable",
                    )

        def stop_stream(self):
            global flag_vc
            if flag_vc:
                flag_vc = False
                if self.stream is not None:
                    self.stream.abort()
                    self.stream.close()
                    self.stream = None

        def audio_callback(
            self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
        ):
            """
            音频处理
            """
            global flag_vc
            start_time = time.perf_counter()
            # Promote the audio thread to real-time priority on its first run
            # (this code path is the only one that executes on that thread).
            if not getattr(self, "_prio_set", False):
                self._prio_set = True
                dbg.note("audio thread: " + rt_priority.boost_current_thread(RT_PRIO))
            indata = librosa.to_mono(indata.T)
            if self.gui_config.threhold > -60:
                indata = np.append(self.rms_buffer, indata)
                rms = librosa.feature.rms(
                    y=indata, frame_length=4 * self.zc, hop_length=self.zc
                )[:, 2:]
                self.rms_buffer[:] = indata[-4 * self.zc :]
                indata = indata[2 * self.zc - self.zc // 2 :]
                db_threhold = (
                    librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
                )
                for i in range(db_threhold.shape[0]):
                    if db_threhold[i]:
                        indata[i * self.zc : (i + 1) * self.zc] = 0
                indata = indata[self.zc // 2 :]
            # Input EQ (shapes the mic before conversion) + stash for waterfall.
            indata = self.eq_in.process(indata)
            self._wf_in_block = indata
            self.input_wav[: -self.block_frame] = self.input_wav[
                self.block_frame :
            ].clone()
            self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
                self.config.device
            )
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                self.block_frame_16k :
            ].clone()
            # input noise reduction and resampling
            if self.gui_config.I_noise_reduce:
                self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                    self.block_frame :
                ].clone()
                input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
                input_wav = self.tg(
                    input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
                ).squeeze(0)
                input_wav[: self.sola_buffer_frame] *= self.fade_in_window
                input_wav[: self.sola_buffer_frame] += (
                    self.nr_buffer * self.fade_out_window
                )
                self.input_wav_denoise[-self.block_frame :] = input_wav[
                    : self.block_frame
                ]
                self.nr_buffer[:] = input_wav[self.block_frame :]
                self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                    self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
                )[160:]
            else:
                self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = (
                    self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[
                        160:
                    ]
                )
            # infer
            if self.function == "vc":
                infer_wav = self.rvc.infer(
                    self.input_wav_res,
                    self.block_frame_16k,
                    self.skip_head,
                    self.return_length,
                    self.gui_config.f0method,
                )
                if self.resampler2 is not None:
                    infer_wav = self.resampler2(infer_wav)
            elif self.gui_config.I_noise_reduce:
                infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
            else:
                infer_wav = self.input_wav[self.extra_frame :].clone()
            # output noise reduction
            if self.gui_config.O_noise_reduce and self.function == "vc":
                self.output_buffer[: -self.block_frame] = self.output_buffer[
                    self.block_frame :
                ].clone()
                self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
                infer_wav = self.tg(
                    infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
                ).squeeze(0)
            # volume envelop mixing
            if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
                if self.gui_config.I_noise_reduce:
                    input_wav = self.input_wav_denoise[self.extra_frame :]
                else:
                    input_wav = self.input_wav[self.extra_frame :]
                rms1 = librosa.feature.rms(
                    y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                    frame_length=4 * self.zc,
                    hop_length=self.zc,
                )
                rms1 = torch.from_numpy(rms1).to(self.config.device)
                rms1 = F.interpolate(
                    rms1.unsqueeze(0),
                    size=infer_wav.shape[0] + 1,
                    mode="linear",
                    align_corners=True,
                )[0, 0, :-1]
                rms2 = librosa.feature.rms(
                    y=infer_wav[:].cpu().numpy(),
                    frame_length=4 * self.zc,
                    hop_length=self.zc,
                )
                rms2 = torch.from_numpy(rms2).to(self.config.device)
                rms2 = F.interpolate(
                    rms2.unsqueeze(0),
                    size=infer_wav.shape[0] + 1,
                    mode="linear",
                    align_corners=True,
                )[0, 0, :-1]
                rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
                infer_wav *= torch.pow(
                    rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate)
                )
            # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
            conv_input = infer_wav[
                None, None, : self.sola_buffer_frame + self.sola_search_frame
            ]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(
                F.conv1d(
                    conv_input**2,
                    torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
                )
                + 1e-8
            )
            if sys.platform == "darwin":
                _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
            infer_wav = infer_wav[sola_offset:]
            if "privateuseone" in str(self.config.device) or not self.gui_config.use_pv:
                infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
                infer_wav[: self.sola_buffer_frame] += (
                    self.sola_buffer * self.fade_out_window
                )
            else:
                infer_wav[: self.sola_buffer_frame] = phase_vocoder(
                    self.sola_buffer,
                    infer_wav[: self.sola_buffer_frame],
                    self.fade_out_window,
                    self.fade_in_window,
                )
            self.sola_buffer[:] = infer_wav[
                self.block_frame : self.block_frame + self.sola_buffer_frame
            ]
            # Output EQ (shapes the converted voice) + output scope, then
            # fan the mono result out to the device channels.
            out_mono = infer_wav[: self.block_frame].cpu().numpy()
            out_mono = self.eq_out.process(out_mono)
            self._wf_out_block = out_mono
            self._wf_seq += 1
            outdata[:] = np.repeat(
                out_mono[:, None], self.gui_config.channels, axis=1
            )
            total_time = time.perf_counter() - start_time
            if flag_vc:
                self.window["infer_time"].update(int(total_time * 1000))
            # Feed the debug logger: status carries sounddevice's xrun flags,
            # out_peak detects digital clipping at the device boundary.
            dbg.tick(
                total_time * 1000.0,
                status=status,
                out_peak=float(np.abs(outdata).max()),
                sola_offset=int(sola_offset),
            )

        def update_devices(self, hostapi_name=None):
            """获取设备列表"""
            global flag_vc
            flag_vc = False
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            self.hostapis = [hostapi["name"] for hostapi in hostapis]
            if hostapi_name not in self.hostapis:
                hostapi_name = self.hostapis[0]
            self.input_devices = [
                d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices = [
                d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.input_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]

        def set_devices(self, input_device, output_device):
            """设置输出设备"""
            sd.default.device[0] = self.input_devices_indices[
                self.input_devices.index(input_device)
            ]
            sd.default.device[1] = self.output_devices_indices[
                self.output_devices.index(output_device)
            ]
            printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
            printt("Output device: %s:%s", str(sd.default.device[1]), output_device)

        def get_device_samplerate(self):
            return int(
                sd.query_devices(device=sd.default.device[0])["default_samplerate"]
            )

        def get_device_channels(self):
            max_input_channels = sd.query_devices(device=sd.default.device[0])[
                "max_input_channels"
            ]
            max_output_channels = sd.query_devices(device=sd.default.device[1])[
                "max_output_channels"
            ]
            return min(max_input_channels, max_output_channels, 2)

    gui = GUI()
