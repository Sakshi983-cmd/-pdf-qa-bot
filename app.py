import os
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from googletrans import Translator

# Page config
st.set_page_config(page_title="SamjhoPDF", layout="wide")
st.title("📄 SamjhoPDF — Understand Any PDF in Hindi & English")

# File upload
uploaded_files = st.file_uploader("📁 Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    pdf_texts = {}
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            texts = [page.extract_text() or "" for page in pdf.pages]
            clean_text = "\n".join([t.strip() for t in texts if t.strip()])
            pdf_texts[uploaded_file.name] = clean_text

    st.success(f"{len(pdf_texts)} PDF(s) processed.")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_chunks = []
    chunk_to_doc = []

    for doc_name, text in pdf_texts.items():
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        chunk_to_doc.extend([doc_name] * len(chunks))

    # Embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(all_chunks, show_progress_bar=False)

    # FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings))

    # QA pipeline
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    translator = Translator()

    # User query
    query = st.text_input("💬 Ask a question based on PDFs (e.g., 'Isme kya likha hai?' or 'Conclusion kya hai?')")

    if query:
        query_embedding = embedder.encode([query])
        distances, indices = index.search(np.array(query_embedding), 5)
        merged_context = " ".join([all_chunks[idx] for idx in indices[0]])

        result = qa_pipeline(question=query, context=merged_context)

        # Smart prompt suggestion
        if len(query.strip()) < 10:
            st.warning("❗ Your question seems short. Try asking: 'What is the main goal of this paper?' or 'Who created this document?'")

        # Output
        st.markdown(f"📄 **From:** {', '.join(set(chunk_to_doc[idx] for idx in indices[0]))}")
        st.markdown(f"✅ **Answer:** {result['answer']}")
        st.markdown(f"✏️ *Confidence Score:* {result['score']:.4f}")
        st.markdown(f"📌 **Context Preview:** {merged_context[:500]}...")

        # Summary
        with st.expander("📝 Summary | सारांश"):
            summary_en = summarizer(merged_context[:1000], max_length=150, min_length=40)[0]['summary_text']
            st.write("📘 English:", summary_en)
            summary_hi = translator.translate(summary_en, dest='hi').text
            st.write("📗 हिंदी:", summary_hi)

        # Conclusion
        with st.expander("📌 Conclusion | निष्कर्ष"):
            conclusion_prompt = f"Extract the conclusion from this document:\n{merged_context[:1000]}"
            conclusion = qa_pipeline(question="What is the conclusion?", context=merged_context)
            st.write("📘 English:", conclusion['answer'])
            conclusion_hi = translator.translate(conclusion['answer'], dest='hi').text
            st.write("📗 हिंदी:", conclusion_hi)

        # Tone detection (simple prompt)
        with st.expander("🎭 Tone Detection | दस्तावेज़ का मूड"):
            tone_prompt = f"Analyze the tone of this document: {merged_context[:1000]}"
            tone_result = qa_pipeline(question="What is the tone of this document?", context=merged_context)
            st.write("🧠 Tone:", tone_result['answer'])

        # Keyword cloud
        with st.expander("📊 Keyword Cloud | प्रमुख शब्द"):
            wordcloud = WordCloud(width=800, height=400).generate(merged_context)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

        # Footer
        st.markdown("---")
        st.markdown("🧠 Made with ❤️ by Sakshi Tiwari")

