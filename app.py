# ==========================================================
# 🦟 ENHANCING MALARIA DETECTION USING CNN AND RAG (One Cell)
# Google Drive version — Groq API in backend (no UI key)
# Default model: llama-3.3-70b-versatile with fallback to openai/gpt-oss-120b
# Enhanced with treatment advice, references, and precautions
# ==========================================================

# -----------------------------
# ✅ Install dependencies
# -----------------------------
!pip install -q faiss-cpu gradio openpyxl joblib scikit-learn tensorflow opencv-python pandas requests

# -----------------------------
# ✅ Imports
# -----------------------------
import os, zipfile, joblib, requests, json
import numpy as np
import pandas as pd
import cv2
import faiss
import gradio as gr
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# ✅ Mount Google Drive
# -----------------------------
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = "/content/drive/MyDrive"

# -----------------------------
# ✅ Configuration
# -----------------------------
MODE = "load"          # "train" for first time, "load" for reuse
FORCE_RETRAIN = False  # True = retrain even if artifacts exist

# Paths
ARTIFACT_DIR = f"{DRIVE_ROOT}/malaria_artifacts"
MODEL_PATH       = f"{ARTIFACT_DIR}/malaria_model.h5"
VECTORIZER_PATH  = f"{ARTIFACT_DIR}/vectorizer.joblib"
CSV_PATH         = f"{ARTIFACT_DIR}/malaria_cases.csv"
FAISS_PATH       = f"{ARTIFACT_DIR}/faiss_index.index"

ZIP_PATH     = f"{DRIVE_ROOT}/archive.zip"
EXTRACT_PATH = "/content/data_extract"
EXCEL_PATH   = f"{DRIVE_ROOT}/Malaria_iIlment_and_Grading_Dataset.xlsx"

IMG_SIZE, SEED, EPOCHS, BATCH_SIZE = 50, 42, 10, 32
DEFAULT_K = 3

# ✅ Updated default Groq model (previous 3.1 model was decommissioned)
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
FALLBACK_GROQ_MODEL = "openai/gpt-oss-120b"

TITLE = "ENHANCING MALARIA DETECTION USING CNN AND RAG"
DESC = (
    "Upload a blood smear image to classify (Parasitized/Uninfected) and/or enter symptoms "
    "to retrieve similar cases via TF-IDF + FAISS. Get AI-powered treatment advice, references, "
    "and precautions based on similar cases. **Research use only. Not medical advice.**"
)

Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# 🔒 Backend secret (your key)
# -----------------------------
# Your backend variable as requested (env var GROQ_API_KEY overrides this)
api_malaria = "YOUR_GROQ_API"

def _get_secret(name: str) -> str | None:
    # 1) Environment overrides
    v = os.environ.get(name)
    if v:
        return v.strip()
    # 2) Optional Colab secrets
    try:
        from google.colab import userdata
        v2 = userdata.get(name)
        if v2:
            return v2.strip()
    except Exception:
        pass
    return None

GROQ_KEY = _get_secret("GROQ_API_KEY") or api_malaria
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# -----------------------------
# ✅ Helper functions
# -----------------------------
def unzip_if_needed(zip_path: str, extract_to: str):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    Path(extract_to).mkdir(parents=True, exist_ok=True)
    if not any(Path(extract_to).rglob("*")):
        print(f"[INFO] Extracting {zip_path} -> {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
    else:
        print(f"[INFO] Already extracted: {extract_to}")

def load_images(root: str, img_size=IMG_SIZE):
    data_path = os.path.join(root, "cell_images", "cell_images")
    folders = [(os.path.join(data_path, "Parasitized"), 1),
               (os.path.join(data_path, "Uninfected"), 0)]
    data, labels = [], []
    for folder, label in folders:
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            data.append(img); labels.append(label)
    X = np.array(data, dtype="float32") / 255.0
    y = to_categorical(np.array(labels), 2)
    print(f"[INFO] Loaded images: X={X.shape}, y={y.shape}")
    return X, y

def build_cnn():
    m = tf.keras.Sequential([
        tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'), tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5), tf.keras.layers.Dense(2, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def train_cnn(X, y):
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=SEED)
    m=build_cnn()
    m.fit(Xtr,ytr,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=0.1,verbose=1)
    print(f"[INFO] Test Acc={m.evaluate(Xte,yte,verbose=0)[1]:.4f}")
    return m

def build_rag_artifacts(excel_path: str):
    df=pd.read_excel(excel_path).fillna("")
    corpus=df.apply(lambda r:' '.join([
        str(r.get('Complaints/Symptoms',"")),
        str(r.get('Outcome',"")),
        str(r.get('Malaria Outcome Interpretation',"")),
        str(r.get('Complicated/ Uncomplicated Malaria Diagnosis',""))
    ]),axis=1).tolist()
    v=TfidfVectorizer(); Xc=v.fit_transform(corpus).toarray().astype('float32')
    idx=faiss.IndexFlatL2(Xc.shape[1]); idx.add(Xc)
    joblib.dump(v,VECTORIZER_PATH); df.to_csv(CSV_PATH,index=False); faiss.write_index(idx,FAISS_PATH)
    print("[INFO] RAG artifacts saved.")
    return v,df,idx

def artifacts_exist():
    return all(os.path.exists(p) for p in [MODEL_PATH,VECTORIZER_PATH,CSV_PATH,FAISS_PATH])

def load_artifacts():
    return load_model(MODEL_PATH), joblib.load(VECTORIZER_PATH), pd.read_csv(CSV_PATH), faiss.read_index(FAISS_PATH)

# -----------------------------
# 🧠 Groq chat helper (backend-only, with fallback)
# -----------------------------
def _groq_chat(messages, model=DEFAULT_GROQ_MODEL, temperature=0.2, max_tokens=800):
    """
    Calls Groq chat completions from the backend using a secret key.
    If primary model fails (e.g., decommissioned), tries FALLBACK_GROQ_MODEL.
    Returns (text, error).
    """
    if not GROQ_KEY:
        return None, "Groq key not configured on backend."

    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    def _post(p):
        return requests.post(GROQ_CHAT_URL, headers=headers, data=json.dumps(p), timeout=30)

    try:
        r = _post(payload)
        if r.status_code != 200:
            # Try fallback once
            err_primary = r.text
            payload["model"] = FALLBACK_GROQ_MODEL
            r2 = _post(payload)
            if r2.status_code != 200:
                return None, f"Groq API error: {r.status_code} {err_primary}"
            data2 = r2.json()
            text2 = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text2.strip() if text2 else None, None

        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return text.strip() if text else None, None

    except Exception as e:
        return None, f"Groq call failed: {e}"

def _build_treatment_prompt(symptoms: str, rows: list[dict]) -> list[dict]:
    """Creates a comprehensive treatment advice prompt with references and precautions."""
    top = rows[:5]  # Use top 5 cases for better context

    def fmt(r):
        d = r.get('__distance__', '')
        try:
            d = f"{float(d):.4f}"
        except Exception:
            d = str(d)
        return (
            f"- **Symptoms:** {r.get('Complaints/Symptoms','N/A')}\n"
            f"  **Diagnosis:** {r.get('Complicated/ Uncomplicated Malaria Diagnosis','N/A')}\n"
            f"  **Outcome:** {r.get('Malaria Outcome Interpretation','N/A')}\n"
            f"  **Treatment/Outcome:** {r.get('Outcome','N/A')}\n"
            f"  **Similarity Score:** {d}"
        )

    cases_text = "\n\n".join(fmt(r) for r in top)

    system = (
        "You are a medical AI assistant specializing in malaria case analysis. "
        "Based on similar historical cases from the database, provide evidence-based treatment recommendations, "
        "clinical references, and safety precautions. "
        "\n\n**IMPORTANT DISCLAIMERS:**\n"
        "- This is for research and educational purposes ONLY\n"
        "- NOT a substitute for professional medical advice\n"
        "- Always consult qualified healthcare providers for diagnosis and treatment\n"
        "- Treatment should be personalized by medical professionals"
    )

    user = (
        f"**Patient Symptoms:**\n{symptoms}\n\n"
        f"**Similar Cases from Database (TF-IDF + FAISS):**\n{cases_text}\n\n"
        "Based on these similar cases, provide a comprehensive analysis with:\n\n"
        "1. **Treatment Recommendations:**\n"
        "   - Common treatment approaches from similar cases\n"
        "   - First-line and alternative medications mentioned\n"
        "   - Typical treatment duration and protocols\n\n"
        "2. **Clinical References:**\n"
        "   - Patterns observed in similar cases\n"
        "   - Severity indicators (complicated vs uncomplicated)\n"
        "   - Expected outcomes based on historical data\n\n"
        "3. **Precautions & Red Flags:**\n"
        "   - Warning signs requiring immediate medical attention\n"
        "   - Risk factors for complications\n"
        "   - Monitoring recommendations\n"
        "   - When to escalate care\n\n"
        "4. **Patient Guidance:**\n"
        "   - Next steps for seeking professional care\n"
        "   - Questions to ask healthcare provider\n"
        "   - Important considerations\n\n"
        "Format your response clearly with headers. Be specific but emphasize this is educational guidance "
        "based on similar cases, not a personal medical prescription."
    )

    return [{"role":"system","content":system},{"role":"user","content":user}]

# -----------------------------
# ✅ Train or Load pipeline
# -----------------------------
tf.random.set_seed(SEED); np.random.seed(SEED)

if MODE=="train" or (MODE=="load" and not artifacts_exist()) or FORCE_RETRAIN:
    print("[INFO] Training pipeline.")
    unzip_if_needed(ZIP_PATH,EXTRACT_PATH)
    X,y=load_images(EXTRACT_PATH)
    model=train_cnn(X,y); model.save(MODEL_PATH)
    print(f"[INFO] Saved CNN -> {MODEL_PATH}")
    vectorizer,df,index=build_rag_artifacts(EXCEL_PATH)
else:
    print("[INFO] Loading saved artifacts.")
    model,vectorizer,df,index=load_artifacts()

# -----------------------------
# ✅ Inference functions
# -----------------------------
def predict_image(img_path:str)->str:
    img=cv2.imread(img_path)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    arr=np.expand_dims(img.astype("float32")/255.0,axis=0)
    pred=model.predict(arr,verbose=0)[0]
    return "Parasitized" if int(np.argmax(pred))==1 else "Uninfected"

def search_cases(symptoms:str,k:int=DEFAULT_K):
    vec=vectorizer.transform([symptoms]).toarray().astype("float32")
    D,I=index.search(vec,k)
    return I[0],D[0]

def format_matched_info(row:pd.Series)->str:
    sev="Complicated" if "Complicated" in str(row.get('Complicated/ Uncomplicated Malaria Diagnosis',"")) else "Uncomplicated"
    return (f"### 🧾 Top Matched Case Details\n"
            f"**Symptoms:** {row.get('Complaints/Symptoms','N/A')}\n\n"
            f"**Diagnosis:** {row.get('Complicated/ Uncomplicated Malaria Diagnosis','N/A')}\n\n"
            f"**Outcome:** {row.get('Malaria Outcome Interpretation','N/A')}\n\n"
            f"**Treatment/Result:** {row.get('Outcome','N/A')}\n\n"
            f"**Suggested Severity:** **{sev}**\n\n")

# -----------------------------
# ✅ Gradio UI (no API key field)
# -----------------------------
def handle_image(img_file):
    if img_file is None:
        return "Please upload an image."
    try:
        return f"🧪 Image Classification: **{predict_image(img_file)}**"
    except Exception as e:
        return f"❌ Error: {e}"

def handle_symptoms(symptoms, top_k, groq_model):
    if not str(symptoms).strip():
        return pd.DataFrame(), "❗ Please enter symptoms.", "—"

    # Search for similar cases
    I,D=search_cases(symptoms.strip(),int(top_k))
    rows=[{**df.iloc[int(i)].to_dict(),"__distance__":float(d)} for i,d in zip(I,D)]
    table=pd.DataFrame(rows)
    matched_md=format_matched_info(df.iloc[int(I[0])])

    # Generate comprehensive treatment advice
    llm_text = "—"
    try:
        msgs = _build_treatment_prompt(symptoms, rows)
        text, err = _groq_chat(msgs, model=groq_model, max_tokens=800)
        if text:
            llm_text = f"### 🏥 AI-Generated Treatment Analysis\n\n{text}\n\n"
            llm_text += "---\n\n"
            llm_text += "⚠️ **DISCLAIMER:** This analysis is based on similar historical cases and is for "
            llm_text += "research/educational purposes only. It is NOT medical advice. Always consult qualified "
            llm_text += "healthcare professionals for diagnosis and treatment decisions."
        elif err:
            llm_text = f"(Treatment analysis unavailable: {err})"
    except Exception as e:
        llm_text = f"(Treatment analysis failed: {e})"

    return table, matched_md, llm_text

with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🦟 {TITLE}")
    gr.Markdown(DESC)
    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🔬 Image Classification")
            img_in=gr.Image(label="Upload blood smear image", type="filepath")
            btn_img=gr.Button("Classify Image", variant="primary")
            img_out=gr.Markdown()

        with gr.Column():
            gr.Markdown("## 🩺 Symptom Analysis & Treatment Guidance")
            symptoms_in=gr.Textbox(
                label="Enter Patient Symptoms",
                lines=4,
                placeholder="e.g., fever, chills, headache, body aches, fatigue..."
            )
            topk_in=gr.Slider(3,10,value=5,step=1,label="Number of Similar Cases to Analyze")
            groq_model_in=gr.Textbox(label="Groq Model (backend only)", value=DEFAULT_GROQ_MODEL)
            btn_sym=gr.Button("Generate Treatment Analysis", variant="primary")

    gr.Markdown("---")
    gr.Markdown("## 📊 Analysis Results")

    with gr.Row():
        with gr.Column():
            table_out=gr.Dataframe(interactive=False,label="Similar Cases from Database")
            matched_md=gr.Markdown()

        with gr.Column():
            llm_md=gr.Markdown(label="AI Treatment Analysis")

    btn_img.click(handle_image, img_in, img_out)
    btn_sym.click(handle_symptoms, [symptoms_in, topk_in, groq_model_in], [table_out, matched_md, llm_md])

demo.launch(share=False)