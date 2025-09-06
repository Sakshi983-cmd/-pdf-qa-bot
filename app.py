import streamlit as st
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Title
st.title("📄 PDF Question Answering Bot")

# Upload PDF files
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    pdf_texts = {}

    # Extract text from each PDF
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            texts = [page.extract_text() or "" for page in pdf.pages]
            clean_text = "\n".join([t.strip() for t in texts if t.strip()])
            pdf_texts[uploaded_file.name] = clean_text

    st.success(f"{len(pdf_texts)} PDF(s) processed.")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_chunks = []
    chunk_to_doc = []

    for doc_name, text in pdf_texts.items():
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        chunk_to_doc.extend([doc_name] * len(chunks))

    # Generate embeddings
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
        # Embed query and search similar chunks
        query_embedding = embedder.encode([query])
        distances, indices = index.search(np.array(query_embedding), 5)

        # Merge top chunks into one context
        merged_context = " ".join([all_chunks[idx] for idx in indices[0]])

        # Get answer from QA model
        result = qa_pipeline(question=query, context=merged_context)

        # Display results
        st.markdown(f"📄 **From:** {', '.join(set(chunk_to_doc[idx] for idx in indices[0]))}")
        st.markdown(f"✅ **Answer:** {result['answer']}")
        st.markdown(f"✏️ *Confidence Score:* {result['score']:.4f}")
        st.markdown(f"📌 **Context Preview:** {merged_context[:500]}...")
