import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI
import os

# Page config
st.set_page_config(page_title="Hinglish Translator", page_icon="🌐", layout="centered")

# Define language mappings
language_options = {
    "Hindi": "hin_Deva", "Bengali": "ben_Beng", "Telugu": "tel_Telu",
    "Tamil": "tam_Taml", "Marathi": "mar_Deva", "Gujarati": "guj_Gujr",
    "Malayalam": "mal_Mlym", "Kannada": "kan_Knda", "Oriya": "ory_Orya",
    "Punjabi": "pan_Guru", "Assamese": "asm_Beng", "Maithili": "mai_Deva",
    "Santhali": "sat_Olck", "Kashmiri": "kas_Arab", "Nepali": "nep_Deva",
    "Konkani": "kok_Deva", "Sindhi": "snd_Arab", "Bodo": "brx_Deva"
}

# Model path
model_path = r"C:\Users\hp\model_cache\models--ai4bharat--indictrans2-indic-indic-1B\snapshots\24d732922a0a91d0998d5568e3af37b7a21cd705"

# Load model
with st.spinner("🔄 Loading IndicTrans2 model..."):
    def load_model():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device

try:
    tokenizer, model, device = load_model()
    ip = IndicProcessor(inference=True)
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# UI
st.title("🇮🇳 Hinglish ➜ Indic Language Translator")
st.markdown("🔤 Type Hinglish in ITRANS format (e.g., `mera naam abhisek hai`) and choose a target language.")

input_text = st.text_area("✍️ Enter Hinglish text", height=100)
target_language = st.selectbox("🌐 Select target language", list(language_options.keys()))

if st.button("Translate"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            # Step 1: Transliterate Hinglish to Devanagari
            devanagari = transliterate(input_text.strip(), ITRANS, DEVANAGARI)

            # Step 2: Preprocess
            batch = ip.preprocess_batch([devanagari], src_lang="hin_Deva", tgt_lang=language_options[target_language])

            # Step 3: Generate translation
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            output = model.generate(**inputs, max_length=128)
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)

            # Step 4: Postprocess
            translated = ip.postprocess_batch(decoded, lang=language_options[target_language])[0]

            # Output
            st.subheader("📘 Translation Result")
            st.markdown(f"**Hinglish:** {input_text}")
            st.markdown(f"**Devanagari:** {devanagari}")
            st.markdown(f"**Translated to {target_language}:** {translated}")

        except Exception as e:
            st.error(f"❌ Translation failed: {str(e)}")

st.markdown("---")
st.caption("🚀 Powered by IndicTrans2 & Streamlit")
