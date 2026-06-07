"""Rolling spectrogram waterfall for the RVC realtime GUI.

Split across two threads so the audio path stays cheap:
- the audio callback only stashes the latest block (a reference) + bumps a seq;
- the GUI thread (pinned to P-cores along with the rest of the process) does the
  FFT, log-frequency binning, colormap and PNG encode in `push`/`render_png`.

Layout of the spectrogram array: shape (n_freq, n_time), newest column on the
right, low frequency at the bottom of the rendered image. Magnitudes are dB with
an adaptive ceiling so the display auto-gains to the current signal.
"""

from io import BytesIO

import numpy as np
from PIL import Image


def _build_colormap():
    """256-entry black->blue->cyan->green->yellow->red->white LUT (uint8)."""
    stops = [
        (0.00, (0, 0, 0)),
        (0.18, (0, 0, 140)),
        (0.36, (0, 170, 200)),
        (0.54, (0, 200, 60)),
        (0.72, (230, 230, 0)),
        (0.88, (230, 60, 0)),
        (1.00, (255, 255, 255)),
    ]
    xs = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=np.float64)
    grid = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(grid, xs, cols[:, c]) for c in range(3)], axis=1)
    return lut.astype(np.uint8)


_LUT = _build_colormap()


def blank_png(width, height):
    """Solid dark PNG to reserve the Image element's space before audio starts."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:] = (13, 17, 23)  # matches the GUI's #0d1117 panels
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, "PNG")
    return buf.getvalue()


class Waterfall:
    def __init__(
        self,
        sr,
        fft=2048,
        n_freq=140,
        n_time=260,
        f_lo=50.0,
        f_hi=None,
        out_w=430,
        out_h=170,
        dyn_range=80.0,
    ):
        self.fft = fft
        self.n_freq = n_freq
        self.n_time = n_time
        self.out_w = out_w
        self.out_h = out_h
        self.dyn_range = dyn_range
        self.window = np.hanning(fft).astype(np.float32)
        freqs = np.fft.rfftfreq(fft, 1.0 / sr)
        f_hi = f_hi or min(sr / 2.0, 12000.0)
        edges = np.logspace(np.log10(f_lo), np.log10(f_hi), n_freq + 1)
        # which display band each FFT bin falls into (-1 => outside range)
        self.bin_idx = np.digitize(freqs, edges) - 1
        self.valid = (self.bin_idx >= 0) & (self.bin_idx < n_freq)
        self._bidx = self.bin_idx[self.valid]
        self.spec = np.full((n_freq, n_time), -200.0, dtype=np.float32)
        self.vmax = -60.0

    def push(self, block):
        """Add one column from a 1-D audio block (GUI thread)."""
        x = np.asarray(block, dtype=np.float32)
        if x.size >= self.fft:
            x = x[-self.fft :]
        else:
            x = np.pad(x, (self.fft - x.size, 0))
        mag = np.abs(np.fft.rfft(x * self.window)) / (self.fft / 2)
        db = 20.0 * np.log10(mag + 1e-7)
        col = np.full(self.n_freq, -200.0, dtype=np.float32)
        # peak-pool the FFT bins into the log-spaced display bands
        np.maximum.at(col, self._bidx, db[self.valid])
        self.spec[:, :-1] = self.spec[:, 1:]
        self.spec[:, -1] = col
        self.vmax = max(self.vmax * 0.999, float(col.max()))

    def render_png(self):
        vmax = self.vmax
        vmin = vmax - self.dyn_range
        norm = np.clip((self.spec - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
        idx = (norm * 255).astype(np.uint8)
        rgb = _LUT[idx]  # (n_freq, n_time, 3)
        rgb = rgb[::-1]  # low frequency at the bottom
        img = Image.fromarray(rgb, "RGB").resize(
            (self.out_w, self.out_h), Image.NEAREST
        )
        buf = BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
