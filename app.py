# app.py
import os
import io
import math
import base64
import json
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
from typing import List, Dict, Any

# --- your existing logic ---
from vic import unified_answer  # must return a str

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="Investment Updates Chat", layout="centered")
st.title("Investment Updates Chat")
st.caption("Ask about a company or compare multiple companies. Beep + chart demo are in the sidebar.")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role: "user"/"assistant", content: str}]

if "sound_enabled" not in st.session_state:
    st.session_state.sound_enabled = True  # default ON

# =========================
# SIDEBAR: SETTINGS + DEMOS
# =========================
with st.sidebar:
    st.subheader("Settings")
    st.session_state.sound_enabled = st.checkbox("Play sound on reply", value=st.session_state.sound_enabled)

    st.divider()
    st.subheader("Chart demo")
    st.caption("Renders a simple chart using investment_updates.json (top 10 by latest revenue).")

    if st.button("Show revenue chart"):
        try:
            # Load your dataset (same file vic.py uses)
            with open("investment_updates.json", "r", encoding="utf-8") as f:
                payload = json.load(f)

            def latest_revenue(entry: Dict[str, Any]):
                updates = entry.get("investmentUpdates") or []
                best = None
                for u in updates:
                    try:
                        rev = u["kpis"].get("revenue")
                        yr = u.get("receivedYear")
                        mo = u.get("receivedMonth")
                        if rev is None or yr is None or mo is None:
                            continue
                        key = (int(yr), int(mo))
                        if (best is None) or (key > best[0]):
                            best = (key, float(rev))
                    except Exception:
                        continue
                return best[1] if best else None

            rows = []
            for d in payload.get("data", []):
                name = d.get("companyName", "Unknown")
                rev = latest_revenue(d)
                if rev is not None:
                    rows.append((name, rev))

            rows.sort(key=lambda x: x[1], reverse=True)
            top = rows[:10]

            if not top:
                st.warning("No revenue data found to chart.")
            else:
                # Make matplotlib chart
                names = [t[0] for t in top]
                values = [t[1] for t in top]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(names[::-1], values[::-1])  # reverse for top at top
                ax.set_xlabel("Latest Revenue")
                ax.set_title("Top 10 by Latest Revenue")
                st.pyplot(fig)

                # Also show as an image (PNG) if you want a 'graph image'
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                buf.seek(0)
                st.image(buf, caption="Saved graph image (PNG)", use_container_width=True)

        except FileNotFoundError:
            st.error("investment_updates.json not found in the app directory.")
        except Exception as e:
            st.error(f"Chart error: {e}")

# =========================
# SOUND (beep) HELPER
# =========================
def make_beep_wav_base64(freq=880, ms=150, sr=44100, volume=0.2) -> str:
    """
    Build a tiny WAV in-memory (no SciPy) and return a data: URL base64 string.
    """
    n_samples = int(sr * (ms / 1000.0))
    # Build raw PCM 16-bit
    samples = bytearray()
    for i in range(n_samples):
        s = volume * math.sin(2 * math.pi * freq * (i / sr))
        # clamp and convert to 16-bit signed
        val = int(max(-1.0, min(1.0, s)) * 32767)
        samples += int(val).to_bytes(2, byteorder="little", signed=True)

    # Minimal WAV header
    # riff header
    wav = BytesIO()
    data_size = len(samples)
    fmt_chunk_size = 16
    audio_format = 1      # PCM
    num_channels = 1
    byte_rate = sr * num_channels * 2
    block_align = num_channels * 2
    bits_per_sample = 16

    def write_chunk(tag: bytes, chunk_data: bytes):
        wav.write(tag)
        wav.write(len(chunk_data).to_bytes(4, "little"))
        wav.write(chunk_data)

    # RIFF header
    wav.write(b"RIFF")
    wav.write((36 + data_size).to_bytes(4, "little"))
    wav.write(b"WAVE")

    # fmt  chunk
    fmt = bytearray()
    fmt += (audio_format).to_bytes(2, "little")
    fmt += (num_channels).to_bytes(2, "little")
    fmt += (sr).to_bytes(4, "little")
    fmt += (byte_rate).to_bytes(4, "little")
    fmt += (block_align).to_bytes(2, "little")
    fmt += (bits_per_sample).to_bytes(2, "little")
    write_chunk(b"fmt ", bytes(fmt))

    # data chunk
    write_chunk(b"data", bytes(samples))

    b64 = base64.b64encode(wav.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"

BEEP_SRC = make_beep_wav_base64()

def play_beep():
    if not st.session_state.sound_enabled:
        return
    st.components.v1.html(
        """
        <audio autoplay style="display:none">
          <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
        </audio>
        """,
        height=0,
    )
# =========================
# RENDER PRIOR MESSAGES
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# CHAT INPUT & RESPONSE
# =========================
prompt = st.chat_input("Ask about a company or compare multiple…")
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant reply
    with st.chat_message("assistant"):
        try:
            reply = unified_answer(prompt)
            if not isinstance(reply, str) or not reply.strip():
                reply = "I couldn't generate a response. Try rephrasing your question."
        except Exception as e:
            reply = f"Error: {e}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    play_beep()  # Ding on new assistant message
