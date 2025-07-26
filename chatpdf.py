import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ✅ Load the local text2text-generation model (LaMini-T5)
def load_qa_model():
    return pipeline(
        "text2text-generation",
        model="./models/LaMini-T5",
        tokenizer="./models/LaMini-T5",
        device=0  # 0 for GPU, -1 for CPU
    )

# ✅ Extract text from uploaded PDFs
def load_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# ✅ Chunk the text
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# ✅ Embed the chunks using SentenceTransformer
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="./local_model")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

# ✅ Ask a question using retriever + generation
def ask_question(vectorstore, query):
    print("🔍 Query received:", query)
    retriever_docs = vectorstore.similarity_search(query, k=3)
    print("🔍 Retrieved Docs:", retriever_docs)

    if not retriever_docs:
        return "❌ No relevant information found in the PDF."

    qa_llm = HuggingFacePipeline(pipeline=load_qa_model())
    chain = load_qa_chain(qa_llm, chain_type="stuff")
    result = chain.run(input_documents=retriever_docs, question=query)
    print("💬 LLM Answer:", result)
    return result



# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="📄 Offline PDF Chat")
    st.title("🧠 Ask Questions from Your PDF (100% Offline)")

    # Upload and process PDFs
    pdf_docs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if pdf_docs and st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            text = load_pdf_text(pdf_docs)
            chunks = get_text_chunks(text)
            vectorstore = get_vectorstore(chunks)
            st.session_state.vstore = vectorstore  # 💾 Save vectorstore
            st.success("✅ PDFs processed successfully!")

    # Ask query only if vectorstore exists
    if "vstore" in st.session_state:
        query = st.text_input("Ask a question from your PDFs:")
        if query:
            with st.spinner("Generating answer..."):
                result = ask_question(st.session_state.vstore, query)
                st.write("**Answer:**", result)
    else:
        st.warning("⚠️ Please upload and process PDFs first.")


if __name__ == "__main__":
    main()
