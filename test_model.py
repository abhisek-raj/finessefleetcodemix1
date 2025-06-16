import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from IndicTransToolkit.processor import IndicProcessor
from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI 
import os

# Set page configuration
st.set_page_config(page_title="Hinglish Translator", page_icon="🌐", layout="centered")

# Initialize IndicProcessor
ip = IndicProcessor(inference=True)

# Define language options
language_options = {
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Malayalam": "mal_Mlym",
    "Kannada": "kan_Knda",
    "Oriya": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Assamese": "asm_Beng",
    "Maithili": "mai_Deva",
    "Santhali": "sat_Olck",
    "Kashmiri": "kas_Arab",
    "Nepali": "nep_Deva",
    "Konkani": "kok_Deva",
    "Sindhi": "snd_Arab",
    "Bodo": "brx_Deva"
}

# Local model path
model_path = "./model_cache/models--ai4bharat--indictrans2-indic-indic-dist-320M/snapshots/ffb7582b6d43791f1fb26b2153fc065f2e9ea575"

@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    if not os.path.exists(model_path):
        st.error(f"Model directory not found: {model_path}")
        return None, None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# Load model
tokenizer, model, device = load_model()

if tokenizer is None or model is None:
    st.stop()

# Translation pipeline
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="hin_Deva",
    device=0 if torch.cuda.is_available() else -1
)

# UI Layout
st.title("🇮🇳 Hinglish to Indic Translator")
st.markdown("Enter Hinglish text in ITRANS format (e.g., <span style='color: lightgreen;'>mera naam abhisek hai</span>) and select a target language.", unsafe_allow_html=True)

input_text = st.text_area("Enter Hinglish text", height=100)
target_language = st.selectbox("Select target language", list(language_options.keys()))

if st.button("Translate"):
    if input_text.strip():
        try:
            # Transliterate to Devanagari
            transliterated = transliterate(input_text.strip(), ITRANS, DEVANAGARI)

            # Preprocess input
            batch = ip.preprocess_batch([transliterated], src_lang="hin_Deva", tgt_lang=language_options[target_language])

            # Translate
            output = model.generate(**tokenizer(batch, return_tensors="pt", padding=True).to(device), max_length=128)
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)

            # Postprocess output
            translations = ip.postprocess_batch(decoded, lang=language_options[target_language])

            # Display
            st.subheader("Results")
            st.write(f"*Input (Hinglish)*: {input_text}")
            st.write(f"*Transliterated (Devanagari)*: {transliterated}")
            st.write(f"*Translated ({target_language})*: {translations[0]}")

        except Exception as e:
            st.error(f"Translation failed: {str(e)}")
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.markdown("hinglish to .....")