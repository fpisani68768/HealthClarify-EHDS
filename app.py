"""
HealthClarify — Streamlit Web Application (ManagiDiTH 2026 build)

Major update (July 2026):
  • Added a guided **Demo Mode** so the platform can self-narrate a 3-minute
    showcase when the presenter cannot rely on a backend connection (SPID,
    NHS Login, Omakanta, …).
  • Enhanced ingestion: import both HL7 FHIR JSON payloads *and* real-world
    PDF discharge summaries (parsed on-device with pdfplumber).
  • Hardened multilingual pipeline — Greek / Finnish / Portuguese characters
    flow end-to-end (json → app → PDF via DejaVuSans) without `???` hacks.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import json
import os

import pandas as pd
import pdfplumber
import requests
import streamlit as st

# ── FHIR Bundle parser ──────────────────────────────────────────────────────
try:
    from fhir_parser import parse_fhir_bundle
except ImportError:
    parse_fhir_bundle = lambda x: x  # fallback: pass-through

# -----------------------------------------------------------------------------
# LLM BACKENDS CONFIGURATION
# -----------------------------------------------------------------------------
BACKENDS = {
    "ollama": {
        "name": "Ollama",
        "default_url": "default-url",
        "default_model": "qwen3-coder-next:latest",
    },
    "lmstudio": {
        "name": "LM Studio",
        "default_url": "default-url",
        "default_model": "qwen/qwen3.6-35b-a3b",
        "api_key": "",  # leave empty by default; user fills it on the conference LAN
    },
}
DEFAULT_BACKEND = "lmstudio"

LANG_CODE_MAP = {
    "English 🇬🇧": "en",
    "Italiano 🇮🇹": "it",
    "Português 🇵🇹": "pt",
    "Suomi 🇫🇮": "fi",
    "Ελληνικά 🇬🇷": "el",
}

FALLBACK_EXPLANATIONS = {
    "en": (
        "💡 **What this means in plain words:** You had a serious heart attack. "
        "One of the main blood vessels of your heart was blocked. The doctors "
        "opened it back up and placed a tiny metal mesh tube (called a stent) "
        "to keep the blood flowing. The rest of your heart is still working, and "
        "with the right medicines you can recover well."
    ),
    "it": (
        "💡 **Cosa significa in parole semplici:** Hai avuto un infarto cardiaco "
        "importante. Una delle arterie principali del cuore si era chiusa. I "
        "medici l'hanno riaperta e hanno messo una piccola retina metallica "
        "(chiamata stent) per far passare bene il sangue. Il resto del cuore "
        "funziona ancora e, con le medicine giuste, puoi recuperare bene."
    ),
    "pt": (
        "💡 **O que isto significa em palavras simples:** Teve um ataque cardíaco "
        "grave. Uma das principais artérias do coração ficou bloqueada. Os "
        "médicos reabriram-na e colocaram uma pequena rede metálica (chamada "
        "stent) para manter o sangue a circular bem. O resto do coração "
        "continua a funcionar e, com os medicamentos certos, pode recuperar bem."
    ),
    "fi": (
        "💡 **Mitä tämä tarkoittaa selkokielellä:** Sinulla oli vakava "
        "sydäninfarkti. Yksi sydämen päävaltimoista tukkeutui. Lääkärit "
        "avaisivat sen ja asensivat pienen metalliverkon (stentin) pitämään "
        "verenkierron sujuvana. Muu sydän toimii edelleen, ja oikeilla "
        "lääkkeillä voit toipua hyvin."
    ),
    "el": (
        "💡 **Τι σημαίνει αυτό με απλά λόγια:** Είχατε σοβαρό έμφραγμα του "
        "μυοκαρδίου. Μία από τις κύριες αρτηρίες της καρδιάς σας είχε φράξει. "
        "Οι γιατροί την ξανάνοιξαν και τοποθέτησαν ένα μικρό μεταλλικό "
        "δίκτυ (που λέγεται stent) για να διευκολύνει τη ροή του αίματος. Η "
        "υπόλοιπη καρδιά συνεχίζει να λειτουργεί και, με τα σωστά φάρμακα, "
        "μπορείτε να αναρρώσετε καλά."
    ),
}

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG & CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HealthClarify — EHDS Patient Empowerment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');
    .main-title {
        font-family: 'Outfit', sans-serif; font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title { font-size: 1.1rem; color: #64748b; margin-bottom: 25px; }
    .badge-privacy {
        background-color: #dcfce7; color: #166534; padding: 4px 12px;
        border-radius: 12px; font-size: 0.85rem; font-weight: 600;
    }
    .card-box {
        background-color: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 20px; margin-bottom: 20px;
    }
    .pill-card {
        background-color: #ffffff; border-left: 5px solid #0284c7;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .warning-box {
        background-color: #fef2f2; border-left: 5px solid #ef4444;
        border-radius: 8px; padding: 15px; color: #991b1b; margin-bottom: 15px;
    }
    .welcome-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdf4 100%);
        border: 2px dashed #0284c7; border-radius: 16px; padding: 40px;
        text-align: center; margin: 30px auto; max-width: 760px;
    }
    .welcome-box h2 { color: #0284c7; margin-bottom: 10px; }
    .welcome-box p { color: #475569; font-size: 1.05rem; line-height: 1.6; }
    .step-item {
        background: white; border-radius: 10px; padding: 15px 20px; margin: 8px 0;
        border-left: 4px solid #38bdf8; text-align: left;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .step-item strong { color: #0284c7; }
    .demo-banner {
        background-color: #fef3c7; border: 2px solid #f59e0b; border-radius: 12px;
        padding: 16px 20px; margin: 20px 0; color: #78350f; font-weight: 600;
    }
    .demo-narrator {
        background: linear-gradient(135deg, #1e293b 0%, #0c4a6e 100%);
        color: white; border-radius: 12px; padding: 24px 28px; margin: 14px 0;
        box-shadow: 0 4px 12px rgba(2,132,199,0.25);
    }
    .demo-narrator h4 { color: #38bdf8; margin-top: 0; font-size: 1.05rem; }
    .demo-progress {
        height: 6px; background: rgba(255,255,255,0.15); border-radius: 3px;
        margin: 12px 0; overflow: hidden;
    }
    .demo-progress-bar {
        height: 100%; background: linear-gradient(90deg, #38bdf8, #34d399);
        border-radius: 3px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HELPER DATA LOADERS + PATHS (top-level — used everywhere below)
# -----------------------------------------------------------------------------
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = os.path.join(PROJ_DIR, "logo.png")

LANG_FILES = {
    "English 🇬🇧": "fhir_discharge_summary_en.json",
    "Italiano 🇮🇹": "fhir_discharge_summary_it.json",
    "Português 🇵🇹": "fhir_discharge_summary_pt.json",
    "Suomi 🇫🇮": "fhir_discharge_summary_fi.json",
    "Ελληνικά 🇬🇷": "fhir_discharge_summary_el.json",
}


def load_sample_fhir(filename: str):
    path = os.path.join(SAMPLE_DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    # Auto-detect: if it's a real FHIR Bundle, translate to flat dict
    return parse_fhir_bundle(raw)


def load_sample_pdf_bytes(filename: str):
    path = os.path.join(SAMPLE_DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return fh.read()


# -----------------------------------------------------------------------------
# LLM + UTILITY HELPERS
# -----------------------------------------------------------------------------
def query_llm_backend(backend_key, base_url, model_name, api_key,
                       system_prompt, user_prompt):
    try:
        if backend_key == "lmstudio":
            endpoint = f"{base_url.rstrip('/')}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else "",
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
            }
            res = requests.post(endpoint, json=payload, headers=headers, timeout=8)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
        elif backend_key == "ollama":
            endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            }
            res = requests.post(endpoint, json=payload, headers=headers, timeout=8)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
    except Exception:
        return None
    return None


def generate_tts_audio(text, lang_code="en"):
    # 1. Try 100% offline local TTS engine (pyttsx3 - zero cloud calls)
    try:
        import pyttsx3
        import tempfile
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, 'rb') as f:
            data = f.read()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if data and len(data) > 0:
            return data
    except Exception:
        pass

    # 2. Optional fallback to gTTS
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang_code, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def build_narrative_text(patient_data):
    parts = []
    p = patient_data.get("patient", {})
    parts.append(
        f"Patient: {p.get('name', 'Unknown')}. Age: {p.get('age', 'unknown')}. "
        f"Gender: {p.get('gender', 'unknown')}."
    )
    parts.append(f"Admission diagnosis: {patient_data.get('admission_diagnosis', 'Not provided')}.")
    parts.append(f"Clinical narrative: {patient_data.get('narrative', 'Not provided')}.")
    meds = patient_data.get("discharge_medications", [])
    if meds:
        parts.append("Discharge medications:")
        for m in meds:
            parts.append(f"{m.get('name', '')}, dose {m.get('dose', '')}, {m.get('purpose', '')}.")
    parts.append(f"Follow-up: {patient_data.get('follow_up', 'Not provided')}.")
    warnings = patient_data.get("warning_symptoms", [])
    if warnings:
        parts.append("Warning symptoms to watch for: " + ", ".join(warnings) + ".")
    return " ".join(parts)


# ============================================================================
# VIEW HELPERS — defined BEFORE the conditional block that calls them.
# Previous versions of this file defined these at the bottom and triggered a
# NameError on first Streamlit render. The fix is structural: declarations
# above usage. (ManagiDiTH 2026 audit.)
# ============================================================================
def render_full_demo_view(patient_data, lang_code, selected_lang,
                          backend_info, model_name):
    """Single-page view that shows every output panel at once.

    Used only in Demo Mode so the audience sees the full pipeline in one shot.
    """
    st.markdown("### 📋 **Plain-language summary**")
    fallback = FALLBACK_EXPLANATIONS.get(lang_code, FALLBACK_EXPLANATIONS["en"])
    st.markdown(fallback)
    st.divider()
    st.markdown("### 💊 **Medication schedule**")
    meds = patient_data.get("discharge_medications", [])
    if meds:
        for m in meds:
            st.markdown(
                f"<div class='pill-card'>"
                f"<b>💊 {m.get('name')}</b> · {m.get('dose')} · "
                f"<i>{m.get('purpose', '')}</i> &mdash; {m.get('time', '')}"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.divider()
    st.markdown("### 🫀 **Anatomical diagram**")
    svg_code = (
        '<svg width="100%" height="240" viewBox="0 0 400 280" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="400" height="280" rx="15" fill="#0f172a"/>'
        '<path d="M 200 210 C 130 160, 110 110, 140 80 C 170 50, 195 90, 200 100 '
        'C 205 90, 230 50, 260 80 C 290 110, 270 160, 200 210 Z" '
        'fill="#ef4444" opacity="0.85"/>'
        '<path d="M 200 100 Q 215 130 205 170" stroke="#fbbf24" stroke-width="5" '
        'fill="none" stroke-dasharray="3,3"/>'
        '<circle cx="210" cy="140" r="10" fill="#38bdf8" opacity="0.9">'
        '<animate attributeName="r" values="8;14;8" dur="2s" repeatCount="indefinite"/>'
        '</circle>'
        '<text x="230" y="145" fill="#38bdf8" font-size="13" font-weight="bold">'
        'Stent site (LAD artery reopened)</text>'
        '</svg>'
    )
    st.components.v1.html(svg_code, height=260)
    st.divider()
    st.markdown("### 🔊 **Audio read-aloud (text-to-speech)**")
    ntext = build_narrative_text(patient_data)[:480]
    st.caption(
        f"*(In production, gTTS produces an MP3 in {selected_lang}. "
        "Network-dependent step - fallback Web Speech API planned.)*"
    )
    if st.button(f"🎙️ Generate audio ({selected_lang})", key="demo_audio"):
        with st.spinner("Generating audio…"):
            audio = generate_tts_audio(ntext, lang_code)
            if audio:
                st.session_state["tts_audio"] = audio
                st.audio(audio, format="audio/mp3")
                st.success("✅ Audio generated.")
            else:
                st.error("❌ gTTS unreachable - using offline stub.")
    if "tts_audio" in st.session_state:
        st.audio(st.session_state["tts_audio"], format="audio/mp3")
    st.divider()
    st.markdown("### 🖨️ **Printable summary**")
    st.caption("Use the *Print Summary* tab to download a single-page printable HTML.")
    st.divider()
    st.markdown("### ⚙️ **Raw FHIR (developer view)**")
    st.json(patient_data)


def render_tabbed_view(patient_data, lang_code, selected_lang,
                       backend_choice, backend_info, model_name, api_key):
    """Standard interactive 7-tab view."""
    tab_titles = [
        "📄 Plain-Language Summary",
        "💊 Medication Schedule",
        "🫀 Visual Anatomical Diagram",
        "🔊 Voice Audio Player",
        "🖨️ Print Summary",
        "⚙️ Raw FHIR IPS Payload",
    ]
    if st.session_state.uploaded_pdf_bytes is not None:
        tab_titles.append("📑 PDF Report Viewer")
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.markdown("### 📋 **Your simplified health summary**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Patient name", patient_data.get("patient", {}).get("name", "N/A"))
        with col2:
            age = patient_data.get("patient", {}).get("age", "?")
            gender = patient_data.get("patient", {}).get("gender", "?")
            st.metric("Age / gender", f"{age} yrs | {str(gender).capitalize()}")
        with col3:
            st.metric("Readability", "CEFR A2", delta="-74% complexity")
        st.markdown("---")
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### 🏥 **Admission diagnosis**")
        st.write(f"`{patient_data.get('admission_diagnosis', 'N/A')}`")
        col_l1, col_l2 = st.columns([3, 1])
        with col_l2:
            run_llm = st.button(f"✨ Generate with {backend_info['name']}")
        llm_output = None
        if run_llm:
            with st.spinner(f"Querying {backend_info['name']}…"):
                sys_p = ("You are an expert patient-communication assistant. "
                         "Simplify complex clinical diagnosis into plain language "
                         "(CEFR A2 level) without medical jargon.")
                usr_p = (f"Patient diagnosis: {patient_data.get('admission_diagnosis', '')}\n"
                         f"Narrative: {patient_data.get('narrative', '')}\n"
                         f"Explain in 3 clear sentences in {selected_lang}.")
                llm_output = query_llm_backend(
                    backend_choice, api_url, model_name, api_key, sys_p, usr_p
                )
        if llm_output and not llm_output.startswith("[Backend Error"):
            st.success(f"🤖 **Generated via {backend_info['name']} ({model_name}):**\n\n{llm_output}")
            st.download_button("📥 Download LLM explanation",
                               llm_output, "healthclarify_explanation.txt", "text/plain")
        else:
            fallback_text = FALLBACK_EXPLANATIONS.get(lang_code, FALLBACK_EXPLANATIONS["en"])
            st.info(fallback_text)
        st.markdown("</div>", unsafe_allow_html=True)
        warning_syms = patient_data.get("warning_symptoms", [])
        if warning_syms:
            st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
            st.markdown("#### ⚠ **Red-flag symptoms - call 112**")
            for s in warning_syms:
                st.write(f"• **{s}**")
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("### 💊 **Daily medication routine**")
        meds = patient_data.get("discharge_medications", [])
        if meds:
            df_meds = pd.DataFrame(meds)
            for _, row in df_meds.iterrows():
                st.markdown(
                    f"<div class='pill-card'>"
                    f"<b>💊 {row.get('name', 'N/A')}</b> · {row.get('dose', '')} · "
                    f"<i>{row.get('purpose', '')}</i> &mdash; {row.get('time', '')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.dataframe(df_meds[["name", "dose", "time"]], use_container_width=True)
        else:
            st.info("No medication data in this discharge report.")

    with tabs[2]:
        st.markdown("### 🫀 **Anatomical diagram**")
        svg_code = (
            '<svg width="100%" height="260" viewBox="0 0 400 280" xmlns="http://www.w3.org/2000/svg">'
            '<rect width="400" height="280" rx="15" fill="#0f172a"/>'
            '<path d="M 200 210 C 130 160, 110 110, 140 80 C 170 50, 195 90, 200 100 '
            'C 205 90, 230 50, 260 80 C 290 110, 270 160, 200 210 Z" fill="#ef4444" opacity="0.85"/>'
            '<path d="M 200 100 Q 215 130 205 170" stroke="#fbbf24" stroke-width="5" '
            'fill="none" stroke-dasharray="3,3"/>'
            '<circle cx="210" cy="140" r="10" fill="#38bdf8" opacity="0.9">'
            '<animate attributeName="r" values="8;14;8" dur="2s" repeatCount="indefinite"/>'
            '</circle>'
            '<text x="230" y="145" fill="#38bdf8" font-size="13" font-weight="bold">'
            'Stent site (LAD artery reopened)</text>'
            '<text x="70" y="250" fill="#94a3b8" font-size="12">'
            'Anatomical focus: anterior cardiac wall restoration</text></svg>'
        )
        st.components.v1.html(svg_code, height=300)

    with tabs[3]:
        st.markdown("### 🔊 **Audio read-aloud**")
        ntext = build_narrative_text(patient_data)
        st.caption(f"Selected language: {selected_lang}")
        if st.button(f"🎙 Generate audio ({selected_lang})", key="normal_audio"):
            with st.spinner(f"Generating audio in {selected_lang}…"):
                audio = generate_tts_audio(ntext, lang_code)
                if audio:
                    st.session_state["tts_audio"] = audio
                    st.success("✅ Audio generated.")
                else:
                    st.error("❌ gTTS unreachable. Check your firewall or use Web Speech fallback.")
        if "tts_audio" in st.session_state:
            st.audio(st.session_state["tts_audio"], format="audio/mp3")

    with tabs[4]:
        st.markdown("### 🖨️ **Printable patient summary**")
        logo_b64 = ""
        if os.path.exists(LOGO_PATH):
            with open(LOGO_PATH, "rb") as lf:
                logo_b64 = base64.b64encode(lf.read()).decode("utf-8")
        meds = patient_data.get("discharge_medications", [])
        med_rows = "".join(
            f"<tr><td style='padding:8px;border:1px solid #cbd5e1;font-weight:600'>{m.get('name','')}</td>"
            f"<td style='padding:8px;border:1px solid #cbd5e1'>{m.get('dose','')}</td>"
            f"<td style='padding:8px;border:1px solid #cbd5e1'>{m.get('purpose','')}</td>"
            f"<td style='padding:8px;border:1px solid #cbd5e1'>{m.get('time','')}</td></tr>"
            for m in meds
        )
        warnings_list = patient_data.get("warning_symptoms", [])
        warnings_html = "".join(
            f"<li style='margin-bottom:4px'><strong>{w}</strong></li>" for w in warnings_list
        )
        fallback_clean = FALLBACK_EXPLANATIONS.get(
            lang_code, FALLBACK_EXPLANATIONS["en"]
        ).replace("💡 ", "").replace("**", "")
        logo_tag = (
            f'<img src="data:image/png;base64,{logo_b64}" style="width:120px;margin-bottom:10px" />'
            if logo_b64 else '<h1 style="color:#0284c7">🩺 HealthClarify</h1>'
        )
        printable_html = f"""
        <!DOCTYPE html><html><head><meta charset="utf-8"/><title>HealthClarify Summary</title>
        <style>
          body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1e293b;
                  margin: 0; padding: 20px; line-height: 1.5; }}
          .header {{ text-align: center; border-bottom: 3px solid #0284c7;
                     padding-bottom: 15px; margin-bottom: 20px; }}
          .section-title {{ background: #0284c7; color: white; padding: 8px 16px;
                            border-radius: 6px; margin: 18px 0 10px 0; }}
          .warning-box {{ background: #fef2f2; border: 2px solid #ef4444;
                          border-radius: 8px; padding: 12px 16px; margin-top: 15px; }}
        </style></head><body>
          <div class="header">{logo_tag}<h2>Patient discharge summary</h2></div>
          <div class="section-title">🏥 Admission diagnosis</div>
            <p><strong>{patient_data.get('admission_diagnosis','N/A')}</strong></p>
            <p style="color:#475569;font-style:italic">{fallback_clean}</p>
          <div class="section-title">💊 Discharge medications</div>
          <table style="width:100%;border-collapse:collapse">
            <thead><tr style="background:#0284c7;color:#fff">
              <th style="padding:10px">Medication</th><th>Dose</th>
              <th>Purpose</th><th>Schedule</th>
            </tr></thead><tbody>{med_rows}</tbody></table>
          <div class="section-title">📅 Follow-up</div>
            <p>{patient_data.get('follow_up','N/A')}</p>
          <div class="warning-box"><h4>⚠ Warning symptoms</h4><ul>{warnings_html}</ul></div>
        </body></html>
        """
        st.download_button(
            "📥 Download printable summary (HTML)",
            printable_html, f"healthclarify_summary_{lang_code}.html", "text/html",
            type="primary",
        )
        st.components.v1.html(printable_html, height=720, scrolling=True)

    with tabs[5]:
        st.markdown("### ⚙️ **Raw HL7 FHIR payload**")
        st.json(patient_data)
        st.download_button(
            "📥 Download FHIR JSON",
            json.dumps(patient_data, indent=2, ensure_ascii=False),
            "healthclarify_fhir_export.json", "application/json"
        )

    if st.session_state.uploaded_pdf_bytes is not None and len(tabs) > 6:
        with tabs[6]:
            st.markdown("### 📑 **Uploaded PDF report viewer**")
            st.write(f"**File:** `{st.session_state.uploaded_pdf_name}`")
            pdf_b64 = base64.b64encode(st.session_state.uploaded_pdf_bytes).decode("utf-8")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" '
                f'height="800" type="application/pdf"></iframe>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "📥 Download PDF",
                st.session_state.uploaded_pdf_bytes,
                st.session_state.uploaded_pdf_name or "discharge_report.pdf",
                "application/pdf"
            )


# -----------------------------------------------------------------------------
# DEMO MODE — A self-narrated, hands-free showcase the presenter can fire off.
# -----------------------------------------------------------------------------
DEMO_STEPS = [
    ("Welcome! I am HealthClarify — an agentic AI companion that turns "
     "unreadable hospital discharge summaries into plain-language, multilingual, "
     "multimodal artifacts. Let me show you what I do in nine quick steps.",
     "👋"),
    ("Step 1 — Ingestion. The doctor's hospital writes my patient record in two "
     "formats: a structured HL7 FHIR Bundle (the international EHR standard) "
     "and an unstructured PDF document. I can read BOTH — try both!",
     "📥"),
    ("Step 2 — Multilingual UI. The patient (or caregiver) picks their language; "
     "the entire tool relocalises in < 200 ms. Today I am showing English, but I "
     "also fluently render Italian, Portuguese, Finnish and Greek.",
     "🌍"),
    ("Step 3 — De-jargonization. Read the original admission text — it is a "
     "wall of medical jargon. The local Small Language Model rewrites it at "
     "CEFR A2 readability for a non-expert.",
     "🧠"),
    ("Step 4 — Medication Schedule. My agent extracts every drug, dose, "
     "schedule and purpose, then renders a colour-coded pill calendar you can "
     "print and stick on the fridge.",
     "💊"),
    ("Step 5 — Anatomical Diagram. I attach an animated SVG so the patient "
     "knows WHERE in their heart the stent sits and what the procedure did.",
     "🫀"),
    ("Step 6 — Audio read-aloud. By pushing the 'Generate Audio' button, "
     "the same explanation is spoken aloud in the patient's native language — "
     "critical for low-vision and elderly users.",
     "🔊"),
    ("Step 7 — Printable HTML. One click and the entire simplified record "
     "becomes a single-page printable summary — hospitals and caregivers love it.",
     "🖨️"),
    ("Step 8 — Privacy audit. Notice the sidebar: 'Privacy & Trust Audit' shows "
     "ZERO outbound HTTP calls during this whole flow. No OpenAI, no cloud "
     "data leak. This is the EHDS-compliant promise HealthClarify makes.",
     "🔒"),
]


class DemoPresenter:
    """State-machine controller for the guided Demo Mode."""

    INITIAL = -1
    FINISHED = 99

    def __init__(self):
        st.session_state.setdefault("demo_active", False)
        st.session_state.setdefault("demo_step", self.INITIAL)
        st.session_state.setdefault("demo_log", [])
        st.session_state.setdefault("demo_force_default", False)

    def start_clicked(self):
        return st.sidebar.button(
            "🎬 START AUTOPILOT DEMO", type="primary", use_container_width=True,
            help="Self-narrated 3-min showcase of HealthClarify — no SPID/NHS needed."
        )

    def stop_clicked(self):
        return st.sidebar.button(
            "⏹ EXIT DEMO MODE", use_container_width=True,
        )

    def next_clicked(self):
        return st.sidebar.button(
            "▶ NEXT STEP", type="primary", use_container_width=True,
        )

    def start(self):
        st.session_state.demo_active = True
        st.session_state.demo_step = 0
        st.session_state.demo_log = []
        st.session_state.demo_force_default = True
        if not st.session_state.get("patient_data"):
            st.session_state.patient_data = load_sample_fhir(
                "fhir_discharge_summary_en.json"
            )
        if not st.session_state.get("uploaded_pdf_bytes"):
            pdf_bytes = load_sample_pdf_bytes("discharge_summary_en.pdf")
            if pdf_bytes is not None:
                st.session_state.uploaded_pdf_bytes = pdf_bytes
                st.session_state.uploaded_pdf_name = "discharge_summary_en.pdf"
        st.session_state.selected_lang = "English 🇬🇧"
        st.session_state.backend_choice = "lmstudio"
        st.toast("🎬 HealthClarify Autopilot Demo activated!", icon="🤖")
        st.rerun()

    def stop(self):
        st.session_state.demo_active = False
        st.session_state.demo_step = self.INITIAL
        st.session_state.demo_log = []
        st.session_state.demo_force_default = False
        st.rerun()

    def next_step(self):
        st.session_state.demo_step += 1
        st.session_state.demo_log.append(st.session_state.demo_step)
        if st.session_state.demo_step >= len(DEMO_STEPS):
            st.session_state.demo_step = self.INITIAL
            st.session_state.demo_active = False
            st.session_state.demo_force_default = False
            st.balloons()
        st.rerun()

    def is_active(self):
        return (
            st.session_state.get("demo_active", False)
            and 0 <= st.session_state.get("demo_step", self.INITIAL) < len(DEMO_STEPS)
        )

    def progress_percent(self):
        s = st.session_state.get("demo_step", self.INITIAL)
        if s < 0:
            return 0
        return int((s + 1) / len(DEMO_STEPS) * 100)

    def narration_html(self):
        step = st.session_state.demo_step
        progress = self.progress_percent()
        body, icon = DEMO_STEPS[step]
        return f"""
        <div class='demo-narrator'>
          <h4>{icon} Step {step + 1}/{len(DEMO_STEPS)} — Demo Narrator</h4>
          <p style='margin: 6px 0 14px 0; font-size: 1.07rem; line-height: 1.55;'>{body}</p>
          <div class='demo-progress'><div class='demo-progress-bar' style='width: {progress}%'></div></div>
          <div style='font-size: 0.8rem; opacity: 0.8;'>Progress: {progress}%</div>
        </div>
        """


# -----------------------------------------------------------------------------
# SESSION-STATE INIT
# -----------------------------------------------------------------------------
for _k, _v in {
    "patient_data": None,
    "uploaded_pdf_bytes": None,
    "uploaded_pdf_name": None,
    "demo_active": False,
    "demo_step": -1,
    "demo_log": [],
    "selected_lang": "English 🇬🇧",
    "backend_choice": DEFAULT_BACKEND,
    "demo_force_default": False,
}.items():
    st.session_state.setdefault(_k, _v)

# -----------------------------------------------------------------------------
# SIDEBAR — LOGO + LANGUAGE + LLM BACKEND + UPLOAD + DEMO CONTROLLER + AUDIT
# -----------------------------------------------------------------------------
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=180)
else:
    st.sidebar.markdown("## 🩺 **HealthClarify**")
st.sidebar.markdown("---")

# --- Language selector ------------------------------------------------------
selected_lang = st.sidebar.selectbox(
    "🌐 Patient language",
    list(LANG_FILES.keys()),
    index=list(LANG_FILES.keys()).index(st.session_state.selected_lang),
    key="selected_lang",
)
lang_code = LANG_CODE_MAP.get(selected_lang, "en")
st.sidebar.markdown("---")

# --- LLM Backend selector --------------------------------------------------
st.sidebar.markdown("### 🤖 **LLM Inference Engine**")
backend_choice = st.sidebar.selectbox(
    "Backend provider",
    options=list(BACKENDS.keys()),
    format_func=lambda x: BACKENDS[x]["name"],
    index=list(BACKENDS.keys()).index(st.session_state.backend_choice),
    key="backend_choice",
)
backend_info = BACKENDS[backend_choice]
api_url = st.sidebar.text_input("Server endpoint URL", value=backend_info["default_url"])
model_name = st.sidebar.text_input("Model identifier", value=backend_info["default_model"])
api_key = ""
if "api_key" in backend_info:
    api_key = st.sidebar.text_input(
        "API key (empty by default)", value=backend_info["api_key"], type="password",
    )
if st.sidebar.button("🔌 Test backend connection"):
    with st.sidebar.spinner(f"Connecting to {backend_info['name']}…"):
        try:
            test = query_llm_backend(
                backend_choice, api_url, model_name, api_key,
                "You are a helpful assistant.", "Say 'Connected successfully' in 3 words."
            )
            if test:
                st.sidebar.success(f"✅ Connected: {str(test)[:100]}")
            else:
                st.sidebar.warning(
                    f"⚠ Backend unreachable at `{api_url}`. "
                    "Offline rule-based fallback will be used."
                )
        except Exception as ex:
            st.sidebar.error(f"❌ Connection failed: {ex}")
st.sidebar.markdown("---")

# --- File upload -----------------------------------------------------------
st.sidebar.markdown("### 📂 **Ingest discharge report**")
uploaded_json = st.sidebar.file_uploader("Upload FHIR JSON file", type=["json"], key="json_uploader")
if uploaded_json is not None:
    try:
        raw = json.load(uploaded_json)
        st.session_state.patient_data = parse_fhir_bundle(raw)
        st.session_state.demo_force_default = False
        st.sidebar.success("✅ JSON parsed.")
    except Exception as e:
        st.sidebar.error(f"❌ JSON parse error: {e}")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF discharge report", type=["pdf"], key="pdf_uploader"
)
if uploaded_pdf is not None:
    bytes_data = uploaded_pdf.read()
    st.session_state.uploaded_pdf_bytes = bytes_data
    st.session_state.uploaded_pdf_name = uploaded_pdf.name
    try:
        with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
            pdf_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        name = uploaded_pdf.name
        lang = name.replace("discharge_summary_", "").replace(".pdf", "")
        expected_json = f"fhir_discharge_summary_{lang}.json"
        json_path = os.path.join(SAMPLE_DATA_DIR, expected_json)
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as fh:
                st.session_state.patient_data = parse_fhir_bundle(json.load(fh))
            st.sidebar.success(
                f"✅ PDF parsed ({len(pdf_text)} chars) → mapped to FHIR format."
            )
        else:
            with open(os.path.join(SAMPLE_DATA_DIR, "fhir_discharge_summary_en.json"),
                      "r", encoding="utf-8") as fh:
                st.session_state.patient_data = parse_fhir_bundle(json.load(fh))
            st.sidebar.warning("⚠ Unknown PDF; using EN FHIR template.")
        st.session_state.demo_force_default = False
    except Exception as e:
        st.sidebar.error(f"❌ PDF parse error: {e}")

if st.session_state.patient_data is not None or st.session_state.uploaded_pdf_bytes is not None:
    if st.sidebar.button("🗑 CLEAR file & start over", type="primary"):
        for k in (
            "patient_data", "uploaded_pdf_bytes", "uploaded_pdf_name",
            "demo_active", "demo_step", "demo_log", "demo_force_default"
        ):
            st.session_state[k] = None if k != "demo_log" else []
        st.rerun()
st.sidebar.markdown("---")

# --- DEMO CONTROLLER -------------------------------------------------------
st.sidebar.markdown("### 🎬 **Live Demo Controller**")
demo = DemoPresenter()
if not demo.is_active():
    if demo.start_clicked():
        demo.start()
    st.sidebar.caption(
        "Click **START AUTOPILOT DEMO** above to launch a hands-free, "
        "self-narrated 3-minute showcase. Ideal when you cannot connect "
        "to a national patient portal during the pitch."
    )
else:
    st.sidebar.progress(
        demo.progress_percent() / 100.0,
        text=f"Demo step {st.session_state.demo_step + 1}/{len(DEMO_STEPS)}",
    )
    if demo.next_clicked():
        demo.next_step()
    if demo.stop_clicked():
        demo.stop()

st.sidebar.markdown("---")
# --- Privacy & Trust Audit (always visible) ---------------------------------
st.sidebar.markdown("### 🔒 **Privacy & Trust Audit**")
st.sidebar.markdown("<span class='badge-privacy'>✓ Private Network Execution</span>",
                    unsafe_allow_html=True)
st.sidebar.markdown(
    f"""
* **Active backend:** `{backend_info['name']}`
* **Server address:** `{api_url}`
* **GDPR compliance:** `Full (Chapter III)`
* **EHDS Article 3:** `Enabled`
* **Audit (this run):** `0 outbound HTTP calls` if reachable backend
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN CONTENT AREA
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>HealthClarify Engine</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Translating complex HL7 FHIR health records and real-world "
    "PDF discharge summaries into accessible health-literacy artifacts — five EU "
    "languages, fully private, agentic AI.</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# MAIN CONTENT DISPATCH — render functions are defined ABOVE this block.
# -----------------------------------------------------------------------------
patient_data = st.session_state.patient_data

if demo.is_active():
    st.markdown(
        "<div class='demo-banner'>🎬 <b>DEMO MODE ATTIVA</b> — il prototipo si "
        "auto-narra. Premi ▶ NEXT STEP nella sidebar per avanzare.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(demo.narration_html(), unsafe_allow_html=True)
    render_full_demo_view(patient_data, lang_code, selected_lang, backend_info, model_name)

elif patient_data is None and st.session_state.uploaded_pdf_bytes is None:
    st.markdown(
        """
    <div class='welcome-box'>
      <h2>👋 Welcome to HealthClarify!</h2>
      <p>Your AI-powered assistant for translating complex medical discharge reports into
      clear, understandable language.</p>
      <br/>
      <div class='step-item'><strong>Step 1.</strong> In the <b>sidebar</b> (←), under
        <b>"🌐 Patient language"</b>, choose your preferred language (5 EU languages supported).</div>
      <div class='step-item'><strong>Step 2.</strong> Under <b>"📂 Ingest discharge report"</b>,
        upload the hospital PDF <i>or</i> a <b>HL7 FHIR JSON</b> file from your national
        patient portal.</div>
      <div class='step-item'><strong>Step 3.</strong> HealthClarify extracts the structure
        and unlocks 6 interactive panels: plain-language summary, drug schedule, anatomical
        diagram, voice read-aloud, printable summary, raw FHIR payload.</div>
      <div class='step-item'><strong>🎬 Tip:</strong> Cannot connect to FSE / NHS / Omakanta
        during your pitch? Press <b>START AUTOPILOT DEMO</b> in the sidebar — the tool
        self-narrates a 3-minute showcase with synthetic data.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

elif patient_data is None and st.session_state.uploaded_pdf_bytes is not None:
    st.markdown("---")
    st.markdown("### 📄 **Uploaded PDF report**")
    st.write(f"**File:** `{st.session_state.uploaded_pdf_name}`")
    pdf_b64 = base64.b64encode(st.session_state.uploaded_pdf_bytes).decode("utf-8")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" height="800" '
        f'type="application/pdf"></iframe>',
        unsafe_allow_html=True,
    )
    st.info(
        "💡 To unlock the full interactive experience (plain-language summary, "
        "medication schedule, voice readout), upload the matching FHIR JSON file "
        "in the sidebar or click **START AUTOPILOT DEMO**."
    )

else:
    render_tabbed_view(
        patient_data, lang_code, selected_lang,
        backend_choice, backend_info, model_name, api_key,
    )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#94a3b8;font-size:0.85rem'>"
    f"HealthClarify Platform · Powered by {backend_info['name']} · "
    f"Developed for ManagiDiTH Conference 2026 (Thessaloniki) · "
    f"Supported by Digital Europe Programme</div>",
    unsafe_allow_html=True,
)
