# 🚀 Code-Mixed Language Translation Project

## ⚠️ Important Notes

> **⚠️ Library Compatibility Warning:**  
> This project may **not be compatible with the latest Python versions**.  
> ✅ Recommended: **Python 3.10 or earlier**

> **⚠️ Requirements Alert:**  
> Some libraries are **version-sensitive**. Make sure to install the **compatible versions**.  
> 🛠️ You can try `pip install -r requirements.txt`, but always verify compatibility first.

---

## 📦 Setup Instructions

### ✅ Step 1: Download Model to Local Storage
- Create a Python file, e.g., `download_model.py`
- Add code to download your Hugging Face model to a local path.

> 🧠 This is a **one-time download**, but required for offline/local inference.

---

### 📁 Step 2: Add `IndicTransToolkit`
- Clone or download [IndicTransToolkit](https://github.com/AI4Bharat/IndicTrans2) into your project folder.
- Ensure it sits alongside your main code files (e.g., not nested too deep).

---

### 🔑 Step 3: Add Hugging Face Access Token (Temporarily)
- Add your Hugging Face token **inside** `download_model.py` **just for model download.**
- Example:
  ```python
  os.environ["HF_TOKEN"] = "your_huggingface_token_here"
