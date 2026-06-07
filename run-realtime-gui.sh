#!/usr/bin/env bash
# Launch the RVC realtime GUI with audio routed into a voice-chat app (Discord).
#
#   mic -> (optional denoise) -> RVC GUI (input: "pulse")
#   RVC GUI output ("pulse") -> $RVC_SINK_NAME -> $RVC_VIRTUAL_MIC
#   voice-chat app input      =  "$RVC_VIRTUAL_MIC"
#
# In the GUI choose Input Device = "pulse" and Output Device = "pulse".
#
# All machine-specific names (audio card, mic, sink) and realtime tuning live
# in .markscomp.env. Edit that file for your own hardware; this script reads it.
set -e
cd "$(dirname "$0")"

# When launched from the desktop icon (no terminal attached), log to a file
# so failures are still inspectable. Interactive runs keep console output.
if [ ! -t 1 ]; then
  exec >/tmp/rvc_gui.log 2>&1
fi

# --- Load machine-specific config -------------------------------------------
ENV_FILE="${RVC_ENV_FILE:-.markscomp.env}"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
else
  echo "WARNING: $ENV_FILE not found; using built-in defaults." >&2
fi

# Fallbacks so the script still runs if the env file is missing/incomplete.
RVC_CARD_PROFILE="${RVC_CARD_PROFILE:-output:analog-stereo+input:analog-stereo}"
RVC_SINK_NAME="${RVC_SINK_NAME:-RVC_to_Discord}"
RVC_VIRTUAL_MIC="${RVC_VIRTUAL_MIC:-RVC_Virtual_Mic}"
RVC_CPU_AFFINITY="${RVC_CPU_AFFINITY:-1}"
RVC_RT_PRIO="${RVC_RT_PRIO:-20}"
RVC_DEBUG="${RVC_DEBUG:-1}"
RVC_DEBUG_LOG="${RVC_DEBUG_LOG:-/tmp/rvc_debug.log}"
RVC_DEBUG_INTERVAL="${RVC_DEBUG_INTERVAL:-1.0}"

# 1. Force the audio interface onto its analog profile (not digital/IEC958).
if [ -n "$RVC_CARD" ]; then
  pactl set-card-profile "$RVC_CARD" "$RVC_CARD_PROFILE" 2>/dev/null || true
fi

# 2. Virtual cable sink for RVC output.
if ! pactl list short sinks | grep -q "\\b${RVC_SINK_NAME}\\b"; then
  pactl load-module module-null-sink sink_name="$RVC_SINK_NAME" \
        sink_properties=device.description="$RVC_SINK_NAME" >/dev/null
fi

# 3. Expose the cable as a REAL microphone so sandboxed apps (snap Discord)
#    list it. Discord hides plain "monitor" sources, but shows this.
if ! pactl list short sources | grep -q "\\b${RVC_VIRTUAL_MIC}\\b"; then
  pactl load-module module-remap-source \
        master="${RVC_SINK_NAME}.monitor" \
        source_name="$RVC_VIRTUAL_MIC" \
        source_properties=device.description="$RVC_VIRTUAL_MIC" >/dev/null
fi

# 4. Pick the input source: prefer the denoised mic, else the raw analog mic.
if [ -n "$RVC_NOISETORCH_SOURCE" ] && pactl list short sources | grep -qF "$RVC_NOISETORCH_SOURCE"; then
  export PULSE_SOURCE="$RVC_NOISETORCH_SOURCE"
elif [ -n "$RVC_RAW_MIC" ]; then
  export PULSE_SOURCE="$RVC_RAW_MIC"
fi
export PULSE_SINK="$RVC_SINK_NAME"

# --- Real-time debug instrumentation (see infer/lib/rt_debug.py) ---
# Rolling per-second SUMMARY lines + immediate XRUN/OVERBUDGET/CLIP warnings.
# Watch with:  tail -F "$RVC_DEBUG_LOG"
export RVC_DEBUG RVC_DEBUG_LOG RVC_DEBUG_INTERVAL
# Uncomment in .markscomp.env for an accurate GPU-time split (adds cuda syncs):
# RVC_DEBUG_SYNC=1

# --- Real-time CPU scheduling (see infer/lib/rt_priority.py) ---
# Pin to P-cores (auto-detected) and give the audio thread SCHED_FIFO. Your
# ulimits must allow this without sudo (rtprio, audio group).
export RVC_CPU_AFFINITY RVC_RT_PRIO

source .venv/bin/activate
exec python gui_v1.py
