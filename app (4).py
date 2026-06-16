import streamlit as st
import os
import numpy as np
import pandas as pd
import faiss

from pypdf import PdfReader

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Employee Policy Assistant",
    page_icon="📄",
    layout="wide"
)

# ==================================================
# POLICY DOCUMENTS
# ==================================================

policy_documents = {
    "Employee Handbook.pdf":
    """
    This employee handbook outlines general company policies,
    including conduct, dress code, and communication guidelines.
    All employees are expected to read and adhere to these guidelines.
    Failure to comply may result in disciplinary action.
    """,

    "Leave Policy.pdf":
    """
    Employees are entitled to 12 casual leaves and 15 sick leaves annually.
    Vacation leave accrues at 1.5 days per month.
    All leave requests must be submitted through the HR portal
    at least two weeks in advance, except for emergencies.
    """,

    "Travel Policy.pdf":
    """
    Business travel expenses are reimbursed within 30 days of submission.
    All travel must be pre-approved by a department head.
    Accommodation and flight bookings should be made through the designated travel portal.
    """,

    "Work From Home Policy.pdf":
    """
    Employees may work from home two days per week,
    subject to managerial approval.

    A stable internet connection and dedicated workspace are required.

    Remote work requests should be submitted weekly.
    """,

    "Medical Insurance Policy.pdf":
    """
    All full-time employees are covered under the company's
    comprehensive medical insurance plan.

    Dependents can be added to the plan at an additional cost.

    Refer to the insurance handbook for detailed coverage.
    """
}

# ==================================================
# CREATE PDF FILES
# ==================================================

def create_pdf(filename, content):

    c = canvas.Canvas(
        filename,
        pagesize=letter
    )

    text = c.beginText()

    text.setTextOrigin(
        72,
        720
    )

    for line in content.split("\n"):

        text.textLine(line)

    c.drawText(text)

    c.save()


for filename, content in policy_documents.items():

    if not os.path.exists(filename):

        create_pdf(
            filename,
            content
        )

# ==================================================
# LOAD DOCUMENTS
# ==================================================

documents = []

document_stats = []

for filename in policy_documents.keys():

    reader = PdfReader(filename)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:

            text += extracted + "\n"

    documents.append(text)

    document_stats.append(
        {
            "File Name": filename,
            "Pages": len(reader.pages),
            "Characters": len(text),
            "Words": len(text.split())
        }
    )

stats_df = pd.DataFrame(document_stats)

# ==================================================
# CHUNKING
# ==================================================

all_documents_text = "\n\n".join(documents)

fixed_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100
)

fixed_chunks = [
    doc.page_content
    for doc in fixed_splitter.create_documents(
        [all_documents_text]
    )
]

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

recursive_chunks = [
    doc.page_content
    for doc in recursive_splitter.create_documents(
        [all_documents_text]
    )
]

# ==================================================
# EMBEDDING MODEL
# ==================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# ==================================================
# EMBEDDINGS
# ==================================================

@st.cache_resource
def create_embeddings():

    embeddings = embedding_model.encode(
        recursive_chunks,
        show_progress_bar=False
    )

    return embeddings

embeddings = create_embeddings()

# ==================================================
# FAISS INDEX
# ==================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    np.array(
        embeddings,
        dtype=np.float32
    )
)

# ==================================================
# RETRIEVAL FUNCTION
# ==================================================

def retrieve_chunks(
    query,
    k=3
):

    query_embedding = embedding_model.encode(
        [query]
    )

    distances, indices = index.search(
        np.array(
            query_embedding,
            dtype=np.float32
        ),
        k
    )

    chunks = [
        recursive_chunks[i]
        for i in indices[0]
    ]

    return chunks, distances[0]
# ==================================================
# LOAD LLM
# ==================================================

@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-small"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small"
    )

    return tokenizer, model


tokenizer, llm = load_llm()

# ==================================================
# RAG ANSWER
# ==================================================

def generate_rag_answer(
    question,
    top_k=3
):

    retrieved_chunks, scores = retrieve_chunks(
        question,
        top_k
    )

    context = "\n".join(
        retrieved_chunks
    )

    prompt = f"""
Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = llm.generate(
        **inputs,
        max_new_tokens=80
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {
        "answer": answer,
        "chunks": retrieved_chunks,
        "scores": scores
    }

# ==================================================
# SAMPLE RETRIEVAL QUERIES
# ==================================================

sample_queries = [
    "How many casual leaves are available?",
    "Can employees work from home?",
    "How does travel reimbursement work?",
    "Who is covered under medical insurance?"
]

# ==================================================
# EVALUATION DATA
# ==================================================

evaluation_data = [

    {
        "question":
        "How many casual leaves can I take annually?",

        "expected":
        "12"
    },

    {
        "question":
        "What is the policy for sick leave?",

        "expected":
        "15"
    },

    {
        "question":
        "When should leave requests be submitted?",

        "expected":
        "two weeks"
    },

    {
        "question":
        "How many days per week can I work from home?",

        "expected":
        "two"
    },

    {
        "question":
        "Who approves travel?",

        "expected":
        "department head"
    },

    {
        "question":
        "What insurance is provided?",

        "expected":
        "medical"
    }
]

# ==================================================
# EVALUATION FUNCTION
# ==================================================

def run_evaluation():

    results = []

    correct = 0

    for item in evaluation_data:

        question = item["question"]

        expected = item["expected"]

        response = generate_rag_answer(
            question
        )

        answer = response["answer"]

        match = (
            expected.lower()
            in answer.lower()
        )

        if match:
            correct += 1

        results.append(
            {
                "Question":
                question,

                "Expected":
                expected,

                "Response":
                answer,

                "Match":
                match
            }
        )

    accuracy = (
        correct
        /
        len(evaluation_data)
    ) * 100

    return (
        pd.DataFrame(results),
        accuracy
    )

# ==================================================
# DASHBOARD METRICS
# ==================================================

total_documents = len(
    document_stats
)

total_pages = sum(
    x["Pages"]
    for x in document_stats
)

total_words = sum(
    x["Words"]
    for x in document_stats
)

total_characters = sum(
    x["Characters"]
    for x in document_stats
)

total_chunks = len(
    recursive_chunks
)

total_vectors = index.ntotal

embedding_dimension = dimension

# ==================================================
# SIDEBAR MENU
# ==================================================

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📚 Documents",
        "✂️ Chunk Explorer",
        "🧠 Embeddings",
        "🗄️ Vector Database",
        "🔍 Retrieval Demo",
        "🤖 Policy Assistant",
        "📊 Evaluation"
    ]
)
# ==================================================
# DASHBOARD
# ==================================================

if menu == "🏠 Dashboard":

    st.title("📄 Employee Policy Assistant")

    st.markdown(
        """
        Intelligent Employee Policy Assistant powered by
        RAG (Retrieval-Augmented Generation),
        FAISS Vector Search and FLAN-T5.
        """
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Documents",
        total_documents
    )

    c2.metric(
        "Chunks",
        total_chunks
    )

    c3.metric(
        "Vectors",
        total_vectors
    )

    c4.metric(
        "Embedding Dim",
        embedding_dimension
    )

    st.divider()

    left, right = st.columns([2, 1])

    with left:

        st.subheader(
            "System Overview"
        )

        st.info(
            f"""
            • {total_documents} policy documents loaded

            • {total_chunks} chunks generated

            • FAISS index built successfully

            • {total_vectors} vectors stored

            • Embedding dimension:
            {embedding_dimension}

            • FLAN-T5 integrated
            """
        )

    with right:

        st.subheader(
            "Pipeline"
        )

        st.success(
            """
            PDFs
              ↓
            Chunking
              ↓
            Embeddings
              ↓
            FAISS
              ↓
            Retrieval
              ↓
            LLM
              ↓
            Answer
            """
        )

# ==================================================
# DOCUMENTS PAGE
# ==================================================

elif menu == "📚 Documents":

    st.title("📚 Documents")

    st.dataframe(
        stats_df,
        use_container_width=True
    )

    selected_doc = st.selectbox(
        "Select Document",
        list(policy_documents.keys())
    )

    st.subheader(
        "Document Preview"
    )

    st.info(
        policy_documents[selected_doc]
    )

# ==================================================
# CHUNK EXPLORER
# ==================================================

elif menu == "✂️ Chunk Explorer":

    st.title("✂️ Chunk Explorer")

    tab1, tab2 = st.tabs(
        [
            "Fixed Chunking",
            "Recursive Chunking"
        ]
    )

    with tab1:

        st.write(
            f"Total Chunks: {len(fixed_chunks)}"
        )

        chunk_number = st.slider(
            "Fixed Chunk Number",
            1,
            len(fixed_chunks),
            1
        )

        st.code(
            fixed_chunks[
                chunk_number - 1
            ]
        )

    with tab2:

        st.write(
            f"Total Chunks: {len(recursive_chunks)}"
        )

        chunk_number = st.slider(
            "Recursive Chunk Number",
            1,
            len(recursive_chunks),
            1
        )

        st.code(
            recursive_chunks[
                chunk_number - 1
            ]
        )

# ==================================================
# EMBEDDINGS PAGE
# ==================================================

elif menu == "🧠 Embeddings":

    st.title(
        "🧠 Embedding Explorer"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Embedding Dimension",
        embeddings.shape[1]
    )

    c2.metric(
        "Total Embeddings",
        embeddings.shape[0]
    )

    st.subheader(
        "Sample Chunk"
    )

    st.code(
        recursive_chunks[0]
    )

    st.subheader(
        "Embedding Preview"
    )

    embed_df = pd.DataFrame(
        {
            "Value":
            embeddings[0][:20]
        }
    )

    st.dataframe(
        embed_df,
        use_container_width=True
    )

# ==================================================
# VECTOR DATABASE PAGE
# ==================================================

elif menu == "🗄️ Vector Database":

    st.title(
        "🗄️ FAISS Vector Database"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Stored Vectors",
        index.ntotal
    )

    c2.metric(
        "Embedding Dimension",
        dimension
    )

    c3.metric(
        "Chunks",
        len(recursive_chunks)
    )

    st.divider()

    st.success(
        """
        Vector Database Successfully Built

        • FAISS IndexFlatL2

        • Semantic Search Enabled

        • Embeddings Stored

        • Ready for Retrieval
        """
    )
# ==================================================
# RETRIEVAL DEMO
# ==================================================

elif menu == "🔍 Retrieval Demo":

    st.title("🔍 Semantic Retrieval Demo")

    selected_query = st.selectbox(
        "Choose a Sample Query",
        sample_queries
    )

    if st.button("Retrieve Relevant Chunks"):

        retrieved_chunks, scores = retrieve_chunks(
            selected_query
        )

        st.subheader(
            "Retrieved Chunks"
        )

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            with st.expander(
                f"Chunk {i}"
            ):

                st.write(
                    chunk
                )

                st.caption(
                    f"Distance Score: {scores[i-1]:.4f}"
                )

# ==================================================
# POLICY ASSISTANT
# ==================================================

elif menu == "🤖 Policy Assistant":

    st.title(
        "🤖 Employee Policy Assistant"
    )

    st.markdown(
        """
        Ask questions about:

        • Leave Policy

        • Travel Policy

        • Medical Insurance

        • Work From Home

        • Employee Handbook
        """
    )

    question = st.text_input(
        "Ask your question"
    )

    if st.button(
        "Generate Answer"
    ):

        with st.spinner(
            "Searching policies..."
        ):

            result = generate_rag_answer(
                question
            )

        answer = result["answer"]

        chunks = result["chunks"]

        scores = result["scores"]

        st.subheader(
            "Answer"
        )

        st.success(
            answer
        )

        st.subheader(
            "Sources Used"
        )

        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            with st.expander(
                f"Retrieved Context {i}"
            ):

                st.write(
                    chunk
                )

                st.caption(
                    f"Distance: {scores[i-1]:.4f}"
                )

# ==================================================
# EVALUATION DASHBOARD
# ==================================================

elif menu == "📊 Evaluation":

    st.title(
        "📊 RAG Evaluation"
    )

    st.write(
        """
        Evaluate the Employee Policy Assistant
        using predefined benchmark queries.
        """
    )

    if st.button(
        "Run Evaluation"
    ):

        with st.spinner(
            "Evaluating..."
        ):

            results_df, accuracy = run_evaluation()

        st.subheader(
            "Evaluation Results"
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        st.divider()

        c1, c2 = st.columns(2)

        c1.metric(
            "Accuracy %",
            f"{accuracy:.2f}"
        )

        c2.metric(
            "Total Test Cases",
            len(evaluation_data)
        )

        st.progress(
            min(
                accuracy / 100,
                1.0
            )
        )

        if accuracy >= 80:

            st.success(
                "Excellent Retrieval Performance"
            )

        elif accuracy >= 60:

            st.warning(
                "Moderate Retrieval Performance"
            )

        else:

            st.error(
                "Needs Improvement"
            )