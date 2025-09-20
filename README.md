<div align="center">
  <img src="./assets/avatar.svg" width="120">
</div>

<hr>

<div align="center" style="line-height: 1;">
  
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/github/license/Sakshi983-cmd/-pdf-qa-bot?style=for-the-badge)](./LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-24292F?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Sakshi983-cmd/-pdf-qa-bot)
</div>

<p align="center">
🤖 <b>PDF Question Answering Bot</b><br>
Author: <b>Sakshi Tiwari</b>
</p>

---

## 🦾 Introduction

**PDF Question Answering Bot** is an intelligent assistant that lets users upload PDF files, ask natural language questions, and receive instant, accurate answers powered by advanced AI models.

> Unlock the knowledge inside your documents with one click—fast, scalable, and designed for deep information-seeking.

---

## 🎬 Demo

<p align="center">
  <img src="./assets/demo.png" width="100%">
</p>

---

## 📊 System Workflow

<p align="center">
  <img src="./assets/diagram.png" width="70%">
</p>

Or view as code:

```mermaid
graph LR
    A[PDF Upload] --> B[Preprocessing]
    B --> C[Embedding]
    C --> D[QA Model]
    D --> E[Answer]
    E --> F[User]
```
- **PDF Upload:** User provides the document.
- **Preprocessing:** Text extraction and cleaning.
- **Embedding:** Converts text to machine-understandable vectors.
- **QA Model:** AI model processes queries using document context.
- **Answer:** Response is delivered to the user.

---

## ✨ Features

- ⚡ **Drag & Drop PDF Upload:** Simple, fast document handling.
- 🧠 **AI-powered Question Answering:** Ask any question about your PDFs.
- 🔎 **Contextual Answers:** Returns relevant content snippets with confidence scores.
- 🚦 **Streamlit Interface:** Intuitive and responsive UI.
- 🏗️ **Modular Architecture:** Easily extendable with new models or features.
- 📊 **Confidence & Context Preview:** Transparent outputs for trust and verification.

---

## 🚀 Quick Start

### 1. Environment Setup
- Python >=3.8
- [pip](https://pip.pypa.io/en/stable/)
- (Optional) Docker

```bash
git clone https://github.com/Sakshi983-cmd/-pdf-qa-bot.git
cd -pdf-qa-bot
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

- Upload your PDF file.
- Ask a question in natural language.
- Get instant answers with context and confidence score.

---

## 🛠 Technologies Used

- **Python**
- **Streamlit**
- **HuggingFace Transformers / LangChain**
- **FAISS / Pinecone (Vector DB)**
- **OpenAI / Local LLMs**

---

## 🌟 Star History

<div align="center">
  <a href="https://star-history.com/#Sakshi983-cmd/-pdf-qa-bot&Date">
    <img src="https://api.star-history.com/svg?repos=Sakshi983-cmd/-pdf-qa-bot&type=Date" width="100%">
  </a>
</div>

---

## 📚 Inspired By

README and project inspired by [Alibaba DeepResearch](https://github.com/Alibaba-NLP/DeepResearch).

---

## 📬 Contact

For questions, suggestions, or collaboration:  
**Author:** Sakshi Tiwari  
GitHub: [Sakshi983-cmd](https://github.com/Sakshi983-cmd)  
Email: [your-email@example.com] <!-- Update with your preferred email address -->

---

## 🏷 Citation

If you use this project, please cite as:

```bibtex
@misc{pdfqabot2025,
  author = {Sakshi Tiwari},
  title = {PDF Question Answering Bot},
  year = {2025},
  howpublished = {\url{https://github.com/Sakshi983-cmd/-pdf-qa-bot}}
}
```

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
