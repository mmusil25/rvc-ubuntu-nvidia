"""Real-time debug instrumentation for the RVC realtime GUI.

Audio-callback safe by design: the callback only does cheap arithmetic and a
non-blocking queue put. A background daemon thread performs all file I/O, so
disk latency can never stall the audio thread -- a stall there would itself
produce the xruns/dropouts we are trying to diagnose.

Enable/configure with env vars (all optional):
  RVC_DEBUG=1              turn instrumentation on (default on)
  RVC_DEBUG_LOG=path       text log path (default /tmp/rvc_debug.log)
  RVC_DEBUG_INTERVAL=1.0   seconds between rolling SUMMARY lines
  RVC_DEBUG_SYNC=0         torch.cuda.synchronize() per stage for accurate
                           per-stage GPU timing (small overhead; default off)

Log line vocabulary (greppable):
  ==== ... ====   session banner with the resolved config + per-call budget
  XRUN ...        sounddevice reported an under/overflow this callback
  OVERBUDGET ...  callback processing time exceeded the real-time deadline
  CLIP ...        output sample hit full scale (>= 0.999)
  SUMMARY ...     rolling per-second aggregate (the line to watch)
"""

import os
import queue
import threading
import time


def _truthy(val):
    return str(val).strip().lower() not in ("", "0", "false", "no", "off")


def _now():
    t = time.time()
    lt = time.localtime(t)
    return "%02d:%02d:%02d.%03d" % (
        lt.tm_hour,
        lt.tm_min,
        lt.tm_sec,
        int((t % 1) * 1000),
    )


class _Writer(threading.Thread):
    def __init__(self, path, q):
        super().__init__(daemon=True)
        self.path = path
        self.q = q

    def run(self):
        with open(self.path, "a", buffering=1) as f:
            while True:
                line = self.q.get()
                if line is None:
                    break
                f.write(line + "\n")


class DebugLogger:
    def __init__(self):
        self.enabled = _truthy(os.environ.get("RVC_DEBUG", "1"))
        self.path = os.environ.get("RVC_DEBUG_LOG", "/tmp/rvc_debug.log")
        self.interval = float(os.environ.get("RVC_DEBUG_INTERVAL", "1.0"))
        self.sync = _truthy(os.environ.get("RVC_DEBUG_SYNC", "0"))
        self.budget_ms = 0.0
        self.cb_count = 0
        self.last_cb = None
        self.latest = {}  # most recent SUMMARY snapshot (for future visualizers)
        self._q = queue.Queue(maxsize=8192)
        self._writer = None
        self._reset_window()
        if self.enabled:
            self._writer = _Writer(self.path, self._q)
            self._writer.start()

    # -- internals -----------------------------------------------------------
    def _reset_window(self):
        self.win_start = time.perf_counter()
        self.infer_ms = []
        self.jitter_ms = []
        self.sola = []
        self.stages = []  # (fea, index, f0, model) in ms
        self.n = 0
        self.over = 0
        self.xruns = 0
        self.clips = 0
        self.peak = 0.0

    def _put(self, line):
        try:
            self._q.put_nowait(line)
        except queue.Full:
            pass

    def _emit(self, line):
        if self.enabled:
            self._put("[%s] %s" % (_now(), line))

    @staticmethod
    def _pct(arr, p):
        if not arr:
            return 0.0
        s = sorted(arr)
        i = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
        return s[i]

    # -- public API ----------------------------------------------------------
    def maybe_sync(self):
        """Call before reading a timestamp when accurate stage timing is on."""
        if self.sync:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass

    def note(self, msg):
        """Emit a one-off informational line (e.g. applied CPU priority)."""
        self._emit(msg)

    def banner(self, title, cfg):
        if not self.enabled:
            return
        self.budget_ms = float(cfg.get("budget_ms", 0.0) or 0.0)
        self._emit("==== %s ====" % title)
        for k, v in cfg.items():
            self._emit("  %-18s %s" % (k, v))
        self._emit("=" * 56)
        self._reset_window()
        self.last_cb = None

    def stage(self, fea_s, index_s, f0_s, model_s):
        """Per-inference stage timings (seconds), fed from rtrvc.infer."""
        if self.enabled:
            self.stages.append(
                (fea_s * 1000, index_s * 1000, f0_s * 1000, model_s * 1000)
            )

    def tick(self, infer_ms, status=None, out_peak=0.0, sola_offset=0):
        """Per-callback metrics from the audio thread. Cheap + non-blocking."""
        if not self.enabled:
            return
        now = time.perf_counter()
        self.cb_count += 1
        self.n += 1
        if self.last_cb is not None:
            self.jitter_ms.append((now - self.last_cb) * 1000.0)
        self.last_cb = now
        self.infer_ms.append(infer_ms)
        self.sola.append(sola_offset)
        if out_peak > self.peak:
            self.peak = out_peak
        # immediate, per-event warnings (only fire when something is wrong)
        if status:
            self.xruns += 1
            self._emit("XRUN  %s  (cb #%d)" % (str(status).strip(), self.cb_count))
        if self.budget_ms and infer_ms > self.budget_ms:
            self.over += 1
            self._emit(
                "OVERBUDGET infer=%.1fms budget=%.1fms  (cb #%d)"
                % (infer_ms, self.budget_ms, self.cb_count)
            )
        if out_peak >= 0.999:
            self.clips += 1
            self._emit("CLIP out_peak=%.3f  (cb #%d)" % (out_peak, self.cb_count))
        if now - self.win_start >= self.interval:
            self._flush(now)

    def _flush(self, now):
        dur = now - self.win_start
        if self.n == 0:
            self._reset_window()
            return
        im = self.infer_ms
        mean = sum(im) / len(im)
        snap = {
            "cb_per_s": self.n / dur if dur > 0 else 0.0,
            "infer_mean": mean,
            "infer_p95": self._pct(im, 95),
            "infer_max": max(im),
            "budget_ms": self.budget_ms,
            "over": self.over,
            "over_pct": 100.0 * self.over / self.n,
            "xruns": self.xruns,
            "clips": self.clips,
            "peak": self.peak,
            "jitter_mean": (
                sum(self.jitter_ms) / len(self.jitter_ms) if self.jitter_ms else 0.0
            ),
            "jitter_max": max(self.jitter_ms) if self.jitter_ms else 0.0,
            "sola_mean": sum(self.sola) / len(self.sola) if self.sola else 0.0,
            "sola_max": max(self.sola) if self.sola else 0,
        }
        line = (
            "SUMMARY cb/s=%(cb_per_s).0f infer ms(mean/p95/max)="
            "%(infer_mean).1f/%(infer_p95).1f/%(infer_max).1f budget=%(budget_ms).1f "
            "over=%(over)d(%(over_pct).0f%%) xruns=%(xruns)d clip=%(clips)d "
            "peak=%(peak).3f jitter ms(mean/max)=%(jitter_mean).1f/%(jitter_max).1f "
            "sola(mean/max)=%(sola_mean).0f/%(sola_max).0f" % snap
        )
        if self.stages:
            ns = len(self.stages)
            fea = sum(s[0] for s in self.stages) / ns
            idx = sum(s[1] for s in self.stages) / ns
            f0 = sum(s[2] for s in self.stages) / ns
            mdl = sum(s[3] for s in self.stages) / ns
            snap.update({"fea_ms": fea, "index_ms": idx, "f0_ms": f0, "model_ms": mdl})
            line += " | stage ms fea/index/f0/model=%.1f/%.1f/%.1f/%.1f" % (
                fea,
                idx,
                f0,
                mdl,
            )
            if not self.sync:
                line += " (launch-skewed; set RVC_DEBUG_SYNC=1 for true split)"
        self.latest = snap
        self._emit(line)
        self._reset_window()


_LOGGER = None


def get_logger():
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = DebugLogger()
    return _LOGGER
