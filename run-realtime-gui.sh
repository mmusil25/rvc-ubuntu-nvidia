#!/usr/bin/env bash
# Launch the RVC realtime GUI with audio routed for Discord.
#
#   M-TRACK mic -> NoiseTorch (denoise) -> RVC GUI (input: "pulse")
#   RVC GUI output ("pulse") -> RVC_to_Discord sink -> RVC_Virtual_Mic
#   Discord input            =  "RVC_Virtual_Mic"
#
# In the GUI choose Input Device = "pulse" and Output Device = "pulse".
set -e
cd "$(dirname "$0")"

CARD=alsa_card.usb-M-Audio_M-TRACK_DUO_HD_5000000001-01
# Denoised mic from NoiseTorch (falls back to raw M-TRACK analog if NoiseTorch is off)
NOISETORCH="NoiseTorch Microphone for M-TRACK DUO HD"
RAW_MIC=alsa_input.usb-M-Audio_M-TRACK_DUO_HD_5000000001-01.analog-stereo

# 1. Ensure the M-TRACK is on its ANALOG profile (not digital/IEC958).
pactl set-card-profile "$CARD" output:analog-stereo+input:analog-stereo 2>/dev/null || true

# 2. Virtual cable sink for RVC output.
if ! pactl list short sinks | grep -q '\bRVC_to_Discord\b'; then
  pactl load-module module-null-sink sink_name=RVC_to_Discord \
        sink_properties=device.description=RVC_to_Discord >/dev/null
fi

# 3. Expose the cable as a REAL microphone so sandboxed apps (snap Discord)
#    list it. Discord hides plain "monitor" sources, but shows this.
if ! pactl list short sources | grep -q '\bRVC_Virtual_Mic\b'; then
  pactl load-module module-remap-source \
        master=RVC_to_Discord.monitor \
        source_name=RVC_Virtual_Mic \
        source_properties=device.description=RVC_Virtual_Mic >/dev/null
fi

# 4. Pick the input source: prefer NoiseTorch denoised mic, else raw M-TRACK.
if pactl list short sources | grep -qF "$NOISETORCH"; then
  export PULSE_SOURCE="$NOISETORCH"
else
  export PULSE_SOURCE="$RAW_MIC"
fi
export PULSE_SINK=RVC_to_Discord

source .venv/bin/activate
exec python gui_v1.py
