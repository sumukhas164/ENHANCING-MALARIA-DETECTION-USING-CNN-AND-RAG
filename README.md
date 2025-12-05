# 🦟 Enhancing Malaria Detection Using CNN + RAG

A hybrid **Computer Vision + RAG (Retrieval Augmented Generation)** system that performs:

- **Blood smear image classification** using a custom CNN  
- **Symptom-based case retrieval** using TF-IDF + FAISS  
- **LLM treatment analysis** using Groq API (Llama-3.3-70B) with automatic fallback  
- **Clear UI** built using Gradio (no API key input required on UI)

> ⚠️ **This project is for educational and research purposes only.  
> It is NOT medical advice. Always consult qualified healthcare professionals.**

---

## 🎥 Demo (GIF)


https://github.com/user-attachments/assets/03b68bd3-6387-48b8-838d-2f755391cef8

> 📝 **Reserved space for demo GIF — upload later**
<img width="1762" height="770" alt="Screenshot 2025-12-05 185542" src="https://github.com/user-attachments/assets/44052649-3b64-44c9-bf95-a9c54895b0a9" />
<img width="1694" height="740" alt="Screenshot 2025-12-05 185511" src="https://github.com/user-attachments/assets/97b90444-711b-42fc-a745-819383fe6dcf" />
  
<br><br><br>

---

## 🚀 Features

### 🔬 Image Classification — CNN
- Detects **Parasitized** vs **Uninfected** blood smear images  
- Normalized 50×50 image input  
- Single-shot prediction in milliseconds  

---

### 🩺 Symptom-Based Case Retrieval — RAG
- TF-IDF vectorizer creates embeddings  
- FAISS IndexFlatL2 provides fast similarity search  
- Top-K case retrieval using similarity score  
- Returns symptoms, diagnosis, and outcomes  

---

### 🧠 AI Treatment Guidance — LLM
Powered by **Groq API (Llama-3.3-70B)** with fallback to OpenAI gpt-oss-120b.

Generates:

- Treatment recommendations  
- Severity classification (complicated/uncomplicated)  
- Clinical references  
- Precautions & red flags  
- Patient guidance  

---

### 🖥 Gradio Interface
- Upload blood smear image  
- Enter symptoms  
- View similar cases & similarity scores  
- View AI treatment analysis  

---

## 📁 Project Structure

```
malaria-detection-rag/
│
├── README.md
├── malaria_pipeline_colab.ipynb       # Complete pipeline (train + inference)
├── requirements.txt
│
├── artifacts/                         # Saved model, vectorizer, FAISS index
│   ├── malaria_model.h5
│   ├── vectorizer.joblib
│   ├── malaria_cases.csv
│   ├── faiss_index.index
│
```

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/malaria-detection-rag.git
cd malaria-detection-rag
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install faiss-cpu gradio openpyxl joblib scikit-learn tensorflow opencv-python pandas requests
```

---

## ▶️ How to Run (Colab-Friendly)

1. Open file:  
   **malaria_pipeline_colab.ipynb**

2. Upload datasets:
   - `archive.zip` (blood smear dataset)  
   - `Malaria_iIlment_and_Grading_Dataset.xlsx`

3. Set training mode:
```python
MODE = "train"   # Train from scratch
MODE = "load"    # Use saved artifacts
```

4. Run all cells  
5. Gradio UI launches automatically

---

## 🔧 Environment Variables

Optional (for deployment):

```
GROQ_API_KEY=your_key_here
```

---

## 🤝 Contributing

1. Fork the repo  
2. Create a new branch  
3. Commit your changes  
4. Open a pull request  

---

## 📜 License
MIT License (recommended) — choose your preferred license.

---

## 📩 Contact  
For issues or feature requests, please open a GitHub Issue.

---

## ⭐ Support  
If this project helped you, please ⭐ the repository!

