# 📄 PDF Question Answering Bot

A Streamlit web app that allows users to upload one or more PDF files and ask questions based on their content. It uses advanced NLP models to understand and answer questions accurately.

🔗 **Live App:** [Click here to open](https://9audyrodbv4erbgbsa6neb.streamlit.app)

---

## 🚀 Features

- 📥 **Upload Multiple PDFs** – Drag and drop or select multiple PDF files.
- 🧠 **Intelligent Q&A** – Ask natural language questions based on PDF content.
- 🔍 **Semantic Search** – Uses embeddings for contextual understanding.
- ⚡ **Fast Responses** – Powered by Hugging Face Transformers.
- 🌐 **Streamlit Cloud Deployment** – Accessible from anywhere.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web app framework |
| **pdfplumber** | Extract text from PDFs |
| **SentenceTransformers** (`all-MiniLM-L6-v2`) | Embedding generation for semantic search |
| **Hugging Face Transformers** (`roberta-base-squad2`) | Question answering model |
| **FAISS** | Fast similarity search over embeddings |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/pdf-qa-bot.git
cd pdf-qa-bot

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

