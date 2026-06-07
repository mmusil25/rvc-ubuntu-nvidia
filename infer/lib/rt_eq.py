"""Real-time 3-band EQ for the RVC realtime GUI (input and output streams).

A small cascade of RBJ-cookbook biquads (low-shelf / peaking / high-shelf).
Filter state (``zi``) is carried across audio blocks so re-filtering the next
block continues seamlessly -- without that, every block boundary would inject
a click, i.e. we would create the very artifacts we are trying to remove.

Design notes:
- Gains are in dB; 0 dB == flat. When the EQ is disabled or all three bands
  sit at 0 dB we bypass filtering entirely (no coloration, no cost).
- ``set_gains`` rebuilds the coefficient array but keeps ``zi``; the number of
  sections is constant (3), so the carried state stays valid. The audio thread
  only ever swaps in the new array by reference, which is atomic in CPython.
"""

import numpy as np
from scipy.signal import sosfilt


def _biquad(kind, fs, f0, gain_db, Q):
    """One normalized SOS row [b0,b1,b2,1,a1,a2] (RBJ audio EQ cookbook)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * Q)
    if kind == "peak":
        b0 = 1 + alpha * A
        b1 = -2 * cw
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cw
        a2 = 1 - alpha / A
    elif kind == "lowshelf":
        tsa = 2 * np.sqrt(A) * alpha
        b0 = A * ((A + 1) - (A - 1) * cw + tsa)
        b1 = 2 * A * ((A - 1) - (A + 1) * cw)
        b2 = A * ((A + 1) - (A - 1) * cw - tsa)
        a0 = (A + 1) + (A - 1) * cw + tsa
        a1 = -2 * ((A - 1) + (A + 1) * cw)
        a2 = (A + 1) + (A - 1) * cw - tsa
    elif kind == "highshelf":
        tsa = 2 * np.sqrt(A) * alpha
        b0 = A * ((A + 1) + (A - 1) * cw + tsa)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - tsa)
        a0 = (A + 1) - (A - 1) * cw + tsa
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - tsa
    else:
        raise ValueError(kind)
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


class StreamEQ:
    """3-band shelving/peaking EQ with block-continuous state."""

    # (kind, center frequency Hz, Q) -- low / mid / high
    BANDS = [
        ("lowshelf", 120.0, 0.707),
        ("peak", 1500.0, 1.0),
        ("highshelf", 6000.0, 0.707),
    ]

    def __init__(self, fs):
        self.fs = float(fs)
        self.enabled = False
        self.gains = [0.0, 0.0, 0.0]
        self.sos = self._build()
        self.zi = np.zeros((len(self.BANDS), 2), dtype=np.float64)

    def _build(self):
        rows = [
            _biquad(kind, self.fs, f0, g, Q)
            for (kind, f0, Q), g in zip(self.BANDS, self.gains)
        ]
        return np.array(rows, dtype=np.float64)

    def set_gains(self, enabled, gains):
        self.enabled = bool(enabled)
        self.gains = [float(g) for g in gains]
        self.sos = self._build()

    def process(self, x):
        """Filter a 1-D block in place-equivalent; bypass when flat."""
        if not self.enabled or all(abs(g) < 1e-3 for g in self.gains):
            return x
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(x.dtype, copy=False)
