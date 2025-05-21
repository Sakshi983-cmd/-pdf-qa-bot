import os
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

st.title("📄 PDF Question Answering Bot")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    pdf_texts = {}
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            texts = [page.extract_text() or "" for page in pdf.pages]
            pdf_texts[uploaded_file.name] = "\n".join(texts)

    st.success(f"{len(pdf_texts)} PDF(s) processed.")

    # Chunk text for embeddings
    def chunk_text(text, chunk_size=500):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    all_chunks = []
    chunk_to_doc = []

    for doc_name, text in pdf_texts.items():
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        chunk_to_doc.extend([doc_name] * len(chunks))

    # Load embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(all_chunks, show_progress_bar=False)

    # Create FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings))

    # Load QA pipeline
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # User question input
    query = st.text_input("Ask a question based on PDFs:")

    if query:
        query_embedding = embedder.encode([query])
        distances, indices = index.search(np.array(query_embedding), 3)

        answers = []
        for idx in indices[0]:
            context = all_chunks[idx]
            doc = chunk_to_doc[idx]
            result = qa_pipeline(question=query, context=context)
            answers.append({
                "document": doc,
                "answer": result['answer'],
                "score": result['score'],
                "context_snippet": context[:200]
            })

        for i, res in enumerate(answers, 1):
            st.markdown(f"📄 **From:** {res['document']}")
            st.markdown(f"✅ **Answer:** {res['answer']}")
            st.markdown(f"✏️ *Context snippet:* {res['context_snippet']}...")
