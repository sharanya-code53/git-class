"""
Course PDF Knowledge Assistant - Hackathon Ready
V.R. Renisha, T. Venitha Shree, S. Sandhiya
"""

import os
import tempfile
import io

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

load_dotenv()

# Page config
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")

st.title("📚 Course PDF Knowledge Assistant")
st.markdown(
    "*Upload any course PDF → Ask questions → Get precise answers with diagram (architecture) references*"
)

# Sidebar for API key (secure)
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=st.session_state.api_key,
)
if api_key:
    st.session_state.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key

uploaded_file = st.file_uploader("Upload Course PDF", type="pdf")

diagram_pages = []  # always defined

if uploaded_file and api_key:
    with st.spinner("🔄 Processing your course material..."):
        try:
            # STEP 1: Save uploaded file to temp path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name

            # STEP 2: Try LangChain PyPDFLoader first
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            # If PyPDFLoader fails to give content, fallback to PyMuPDF text
            if not docs or all(not d.page_content.strip() for d in docs):
                # Fallback: manual text extraction via PyMuPDF
                st.warning("PyPDFLoader found no text. Falling back to PyMuPDF text extraction.")
                pdf_doc = fitz.open(temp_path)
                text_docs = []
                for page_index in range(pdf_doc.page_count):
                    page = pdf_doc[page_index]
                    text = page.get_text("text")
                    if text.strip():
                        # Mimic LangChain Document structure minimally
                        text_docs.append(
                            {
                                "page_content": text,
                                "metadata": {"page": page_index + 1},
                            }
                        )
                pdf_doc.close()

                if not text_docs:
                    # Try OCR for scanned PDFs
                    st.warning("No text found. Attempting OCR on images...")
                    pdf_doc_ocr = fitz.open(temp_path)
                    text_docs = []
                    for page_index in range(pdf_doc_ocr.page_count):
                        page = pdf_doc_ocr[page_index]
                        # Render page to image
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_bytes))
                        # OCR the image
                        text = pytesseract.image_to_string(img)
                        if text.strip():
                            text_docs.append(
                                {
                                    "page_content": text,
                                    "metadata": {"page": page_index + 1},
                                }
                            )
                    pdf_doc_ocr.close()

                    if not text_docs:
                        raise ValueError("No text found even with OCR. PDF may be corrupted or have no readable content.")

                # Convert simple dict docs into objects expected by splitter
                class SimpleDoc:
                    def __init__(self, content, meta):
                        self.page_content = content
                        self.metadata = meta

                docs = [SimpleDoc(d["page_content"], d["metadata"]) for d in text_docs]

            # STEP 3: SAFE diagram detection
            try:
                uploaded_file.seek(0)
                pdf_bytes = uploaded_file.read()
                pdf_doc2 = fitz.open(stream=pdf_bytes, filetype="pdf")
                diagram_pages = []
                for page_index in range(pdf_doc2.page_count):
                    page = pdf_doc2[page_index]
                    images = page.get_images(full=True)
                    if images and len(images) > 0:
                        diagram_pages.append(page_index + 1)
                pdf_doc2.close()
            except Exception as e:
                diagram_pages = []
                st.warning(f"Diagram detection skipped: {e}")

            # Display basic info
            col1, col2 = st.columns(2)
            col1.metric("Pages", len(docs))
            col2.metric("Diagrams", len(diagram_pages))
            st.success(
                f"✅ Ready! Loaded {len(docs)} pages with {len(diagram_pages)} diagram/architecture pages"
            )

            # STEP 4: Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = text_splitter.split_documents(docs)

            if not chunks:
                raise ValueError("No text chunks created from PDF (after fallback).")

            # STEP 5: Create FAISS vectorstore (cached)
            @st.cache_resource
            def create_rag_pipeline(_chunks):
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                return FAISS.from_documents(_chunks, embeddings)

            vectorstore = create_rag_pipeline(chunks)
            st.session_state.vectorstore = vectorstore

            # Cleanup temp file
            os.unlink(temp_path)

        except Exception as e:
            st.error(f"❌ Error while processing PDF: {str(e)}")
            st.info("If this is a scanned PDF with only images, OCR is required (not in this prototype).")

# Chat interface
if "vectorstore" in st.session_state:
    st.subheader("💬 Ask questions about your course")

    question = st.text_input(
        "Your question:",
        placeholder="e.g., What is backpropagation?",
    )

    if st.button("Get Answer", type="primary") and question:
        with st.spinner("🤔 Searching course material..."):
            try:
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )
                relevant_docs = retriever.get_relevant_documents(question)

                if not relevant_docs:
                    st.warning("No relevant content found in the PDF for this question.")
                else:
                    context = "\n\n".join(doc.page_content for doc in relevant_docs)

                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    response = llm.invoke(
                        f"""Use ONLY the following course material to answer:

{context}

Question: {question}

Answer concisely and accurately:"""
                    )

                    st.markdown("### 📖 **Answer**")
                    st.write(response.content)

                    if diagram_pages:
                        st.markdown("### 📊 **Diagrams / Architecture Pages**")
                        st.info(f"Pages with diagrams/architectures: **{diagram_pages}**")

                    pages_used = []
                    for d in relevant_docs:
                        page_meta = getattr(d, "metadata", {}).get("page", None)
                        if page_meta is not None:
                            pages_used.append(page_meta)
                    if pages_used:
                        st.caption(f"**Source pages used in answer**: {sorted(set(pages_used))}")

            except Exception as e:
                st.error(f"Answer error: {str(e)}")
else:
    st.info("👈 Enter your OpenAI API key and upload a course PDF to start.")
    st.markdown("---")
    st.markdown(
        """
## 🎯 Demo-ready questions (for any course PDF)
- What is the main topic of this document?
- Summarize the key concepts.
- What are the important definitions?
- Explain [topic] with examples.
"""
    )

st.markdown("---")
st.caption("👩‍💻 V.R. Renisha, T. Venitha Shree, S. Sandhiya | Hackathon Ready")