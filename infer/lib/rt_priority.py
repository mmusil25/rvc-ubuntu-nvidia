"""CPU scheduling helpers for low-latency real-time audio.

Two levers, both effective on an Intel hybrid CPU (P-cores + E-cores) where the
kernel's Thread Director may otherwise park latency-sensitive threads on slow
E-cores:

1. pin_to_pcores()       -- restrict this process (and threads it spawns later)
                            to the performance cores. Needs no privilege.
2. boost_current_thread()-- raise the *calling* thread (intended: the audio
                            callback thread) to SCHED_FIFO real-time priority,
                            falling back to a negative nice value, then to
                            nothing. Honors the user's rtprio/nice ulimits.

Everything degrades gracefully and never raises into the audio path.
"""
import os


def _parse_cpu_list(s):
    out = set()
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def detect_pcores():
    """Return the set of performance-core CPU ids, or None if not a hybrid CPU."""
    try:
        with open("/sys/devices/cpu_core/cpus") as f:
            cpus = _parse_cpu_list(f.read())
        return cpus or None
    except Exception:
        return None


def pin_to_pcores():
    """Pin the current process to the P-cores. Returns the cpu list or None."""
    cpus = detect_pcores()
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return None
    try:
        target = cpus & os.sched_getaffinity(0)
        if not target:
            return None
        os.sched_setaffinity(0, target)
        return sorted(target)
    except Exception:
        return None


def boost_current_thread(rtprio=20):
    """Elevate the calling thread's scheduling priority. Returns a status str."""
    if rtprio and hasattr(os, "SCHED_FIFO") and hasattr(os, "sched_setscheduler"):
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(rtprio)))
            return "SCHED_FIFO prio=%d" % int(rtprio)
        except (PermissionError, OSError, AttributeError, ValueError):
            pass
    try:
        # Fallback: best-effort niceness on the calling thread.
        before = os.nice(0)
        os.nice(-10 - before)  # aim for about -10 if the ulimit allows it
        return "nice=%d (SCHED_FIFO unavailable)" % os.nice(0)
    except Exception:
        return "default scheduling (no elevation permitted)"
