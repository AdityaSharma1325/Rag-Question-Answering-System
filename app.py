"""Ask questions in Streamlit using a LangChain RAG chain with FAISS and Mistral AI."""

# Import os so we can read environment variables from the computer.
import os

# Import hashlib so we can detect when a newly uploaded PDF is different.
import hashlib

# Import Path so we can work with folder paths in a clean and readable way.
from pathlib import Path

# Import Streamlit so we can build a simple web application.
import streamlit as st

# Import load_dotenv so Python can load secret values from the .env file.
from dotenv import load_dotenv

# Import PromptTemplate so LangChain can format the RAG prompt for us.
from langchain_core.prompts import PromptTemplate

# Import RunnablePassthrough so the user's question can flow through the chain.
from langchain_core.runnables import RunnablePassthrough

# Import StrOutputParser so the model response becomes plain text.
from langchain_core.output_parsers import StrOutputParser

# Import FAISS so we can load the saved local vector database.
from langchain_community.vectorstores import FAISS

# Import PyPDFLoader so LangChain can read uploaded PDF files.
from langchain_community.document_loaders import PyPDFLoader

# Import HuggingFaceEmbeddings so we can use the same model used during ingestion.
from langchain_huggingface import HuggingFaceEmbeddings

# Import ChatMistralAI so LangChain can call the Mistral AI chat model.
from langchain_mistralai import ChatMistralAI

# Import RecursiveCharacterTextSplitter so uploaded PDFs can be split into chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add a title at the top of the Streamlit web page.
st.title("RAG Question Answering System")

# Clear Button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Load environment variables from the .env file into this Python program.
load_dotenv()

# Read the Mistral API key from the loaded environment variables.
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Stop the program with a clear message if the Mistral API key is missing.
if not mistral_api_key:
    # Show an error message on the Streamlit page if the API key is missing.
    st.error("MISTRAL_API_KEY is missing. Add it to your .env file.")

    # Stop the Streamlit app so the rest of the code does not run without a key.
    st.stop()

# Store the Mistral API key in the environment where ChatMistralAI expects it.
os.environ["MISTRAL_API_KEY"] = mistral_api_key

# Store the path to the folder where the FAISS database was saved.
VECTORSTORE_FOLDER = Path("vectorstore")

# Store the path to the folder where uploaded PDF files will be saved.
DATA_FOLDER = Path("data")

# Create the data folder if it does not already exist.
DATA_FOLDER.mkdir(exist_ok=True)

# Create the vectorstore folder if it does not already exist.
VECTORSTORE_FOLDER.mkdir(exist_ok=True)

# Create the same HuggingFace embeddings model that was used in ingest.py.
embeddings = HuggingFaceEmbeddings(
    # Use the same sentence-transformers model so query vectors match stored vectors.
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the saved FAISS vector database only if it is not already in session state.
if "vectorstore" not in st.session_state:
    # Load the saved FAISS vector database from the vectorstore folder.
    st.session_state.vectorstore = FAISS.load_local(
        # Tell LangChain which folder contains index.faiss and index.pkl.
        folder_path=str(VECTORSTORE_FOLDER),
        # Pass in the same embeddings object so new questions can be embedded correctly.
        embeddings=embeddings,
        # Allow loading the local pickle file created by our own ingest.py script.
        allow_dangerous_deserialization=True,
    )

# Show a success message when the current FAISS database is available.
st.success(f"FAISS vector database is ready from: {VECTORSTORE_FOLDER}")

# Create the retriever only if it is not already in session state.
if "retriever" not in st.session_state:
    # Create a retriever from the current FAISS vector database.
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(
        # Configure the retriever to return the top 3 most relevant chunks.
        search_kwargs={"k": 3}
    )

# Create a reusable LangChain prompt template for the RAG application.
prompt_template = PromptTemplate(
    # List the variables that LangChain must fill before sending the prompt.
    input_variables=["context", "question"],
    # Write the actual instructions that will be sent to the Mistral model.
    template="""
You are a helpful question-answering assistant.
Answer the question only from the retrieved context below.
Do not use outside knowledge.
If the answer is not present in the context, say:
"I could not find the answer in the provided documents."

Retrieved context:
{context}

Question:
{question}

Answer:
""",
)

# Using the existing prompt template for the conversation history so that the model can refer to previous questions and answers
rewrite_prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="""
You are a query rewriting assistant.

Given the conversation history and latest user question,
rewrite the latest question into a standalone question.

Conversation History:
{history}

Latest Question:
{question}

Standalone Question:
"""
)

# Create the Mistral chat model that LangChain will use inside the RAG chain.
mistral_model = ChatMistralAI(
    # Choose Mistral's latest large chat model for answering questions.
    model="mistral-large-latest",
    # Use zero temperature so answers are more consistent and grounded.
    temperature=0,
)

# Rewriting chain to convert the conversation history and latest question into a standalone question
rewrite_chain = (
    rewrite_prompt
    |mistral_model
    |StrOutputParser()
)

# Define a helper function that turns retrieved documents into one context string.
def format_documents(documents):
    # Join the text from all retrieved chunks with blank lines between them.
    return "\n\n".join(document.page_content for document in documents)


# Add the Conversation awareness so that the system knows what to return when the user asks another question after the first one.
def build_history():
    """
    Convert chat history into plain text
    """
    history = []

    for message in st.session_state.messages:
        history.append(
            f"{message['role']}: {message['content']}"
        )
    return "\n".join(history)


# Define a helper function that builds a RAG chain from the current retriever.
def create_rag_chain(active_retriever):
    # Return a LangChain RAG chain that connects retrieval, prompting, model, and output parsing.
    return (
        # Build the input dictionary required by the prompt template.
        {
            # Retrieve relevant chunks for the question and format them as context.
            "context": active_retriever | format_documents,
            # Pass the original user question through unchanged.
            "question": RunnablePassthrough(),
        }
        # Send the context and question into the prompt template.
        | prompt_template
        # Send the formatted prompt to the Mistral AI model.
        | mistral_model
        # Convert the Mistral chat message into a plain text answer.
        | StrOutputParser()
    )


# Create the RAG chain only if it is not already in session state.
if "rag_chain" not in st.session_state:
    # Build the RAG chain using the current retriever.
    st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)

# Add a PDF uploader that allows users to upload multiple PDF files at once.
uploaded_pdfs = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# Run this block only when at least one PDF has been uploaded.
if uploaded_pdfs:
    # Create an empty list to store each uploaded PDF name and bytes.
    uploaded_pdf_items = []

    # Create a hash object that will represent the full set of uploaded PDFs.
    upload_hash_builder = hashlib.md5()

    # Loop through every uploaded PDF file.
    for uploaded_pdf in uploaded_pdfs:
        # Read the bytes for the current uploaded PDF.
        uploaded_pdf_bytes = uploaded_pdf.getvalue()

        # Add the PDF filename to the hash so file changes are detected clearly.
        upload_hash_builder.update(uploaded_pdf.name.encode("utf-8"))

        # Add the PDF bytes to the hash so content changes are detected clearly.
        upload_hash_builder.update(uploaded_pdf_bytes)

        # Store the PDF filename and bytes for later processing.
        uploaded_pdf_items.append((uploaded_pdf.name, uploaded_pdf_bytes))

    # Create one fingerprint for all uploaded PDFs together.
    uploaded_pdfs_hash = upload_hash_builder.hexdigest()

    # Process the PDFs only when the uploaded set is new or different.
    if st.session_state.get("uploaded_pdfs_hash") != uploaded_pdfs_hash:
        # Show a loading spinner while all uploaded PDFs are prepared.
        with st.spinner("Reading PDFs, creating chunks, and updating FAISS..."):
            # Create an empty list that will hold pages from every uploaded PDF.
            all_pages = []

            # Create an empty list that will hold the uploaded PDF names.
            uploaded_pdf_names = []

            # Loop through every uploaded PDF name and bytes pair.
            for pdf_name, pdf_bytes in uploaded_pdf_items:
                # Create a local path inside the data folder for the current PDF.
                uploaded_pdf_path = DATA_FOLDER / pdf_name

                # Open the local PDF path in binary write mode.
                with open(uploaded_pdf_path, "wb") as pdf_file:
                    # Write the current uploaded PDF bytes to the local file.
                    pdf_file.write(pdf_bytes)

                # Create a LangChain PDF loader for the current PDF file.
                pdf_loader = PyPDFLoader(str(uploaded_pdf_path))

                # Read the current PDF into LangChain Document objects.
                pages = pdf_loader.load()

                # Add the current PDF pages to the combined document list.
                all_pages.extend(pages)

                # Save the current PDF name for display after processing.
                uploaded_pdf_names.append(pdf_name)

            # Create a text splitter for breaking all PDF pages into smaller chunks.
            text_splitter = RecursiveCharacterTextSplitter(
                # Set each chunk to contain about 500 characters.
                chunk_size=500,
                # Keep 50 characters of overlap between chunks for context.
                chunk_overlap=50,
            )

            # Split the combined pages from all PDFs into smaller document chunks.
            chunks = text_splitter.split_documents(all_pages)

            # Create one FAISS database from all chunks across all uploaded PDFs.
            uploaded_vectorstore = FAISS.from_documents(chunks, embeddings)

            # Save the combined FAISS database locally in the vectorstore folder.
            uploaded_vectorstore.save_local(str(VECTORSTORE_FOLDER))

            # Store the combined FAISS database in Streamlit session state.
            st.session_state.vectorstore = uploaded_vectorstore

            # Create a new retriever from the combined FAISS database.
            st.session_state.retriever = uploaded_vectorstore.as_retriever(
                # Configure the retriever to return the top 3 most relevant chunks.
                search_kwargs={"k": 3}
            )

            # Rebuild the RAG chain so it uses the updated multi-PDF retriever.
            st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)

            # Save the combined upload hash so the app knows this PDF set was processed.
            st.session_state.uploaded_pdfs_hash = uploaded_pdfs_hash

            # Save the uploaded PDF names so the app can display the active documents.
            st.session_state.uploaded_pdf_names = uploaded_pdf_names

            # Clear old chat history because the active source documents have changed.
            st.session_state.messages = []

        # Show a success message after all uploaded PDFs are ready for questions.
        st.success(f"Processed {len(uploaded_pdf_names)} PDFs successfully.")
    # Run this block when the same set of uploaded PDFs has already been processed.
    else:
        # Join the active PDF names into one readable string.
        active_pdf_names = ", ".join(st.session_state.uploaded_pdf_names)

        # Show an info message that the current uploaded PDFs are already active.
        st.info(f"Current active PDFs: {active_pdf_names}")

# Create a chat history list in Streamlit session state if it does not already exist.
if "messages" not in st.session_state:
    # Store all previous user questions and AI answers in this list.
    st.session_state.messages = []

# Add a heading before showing the saved conversation.
st.subheader("Conversation")

# Loop through every saved message in the Streamlit session state.
for message in st.session_state.messages:
    # Open a chat bubble using the saved role, such as user or assistant.
    with st.chat_message(message["role"]):
        # Display the saved message text inside the chat bubble.
        st.write(message["content"])

        # Check whether this assistant message has source chunks saved with it.
        if message.get("sources"):
            # Create an expandable area for the source chunks under the answer.
            with st.expander("Source chunks"):
                # Loop through each saved source chunk for this answer.
                for index, source_chunk in enumerate(message["sources"], start=1):
                    # Display a small label for each source chunk.
                    st.markdown(f"**Chunk {index}**")

                    # Display the saved chunk text.
                    st.write(source_chunk["content"])

                    # Display the saved chunk metadata for transparency.
                    metadata = source_chunk["metadata"]
                    pdf_name = Path(
                        metadata.get("source", "Unknown PDF")
                    ).name
                    page_number = metadata.get("page")

                    score = source_chunk.get("score")
                    if page_number is not None:
                        st.caption(
                            f"📄 PDF: {pdf_name} | "
                            f"📑 Page: {page_number + 1} | "
                            f"🎯 Relevance: {score}%"
                        )
                    else:
                        st.caption(
                            f"📄 PDF: {pdf_name} | "
                            f"🎯 Relevance: {score}%"
                        )

                    

# Add a text input box where the user can type a question.
question = st.text_input("Enter your question")

# Add an Ask button that the user can click to run the RAG chain.
ask_button = st.button("Ask")

# Run this block only when the user clicks the Ask button.
if ask_button:
    # Check that the user typed a question before calling the model.
    if question:
        # Show a loading spinner while the RAG chain is working.
        with st.spinner("Searching documents and generating answer..."):
            # Retrieve the top matching document chunks so we can show the sources to the user.
            history = build_history()
            if st.session_state.messages:
                standalone_question = rewrite_chain.invoke(
                    {
                        "history": history,
                        "question": question
                    }
                )
            else:
                standalone_question = question

            #Get source chunks for the standalone question to show the user where the answer came from
            retrieval_results = (
                st.session_state.vectorstore.similarity_search_with_score(
                    standalone_question,
                    k=3
                )
            )
            

            # Invoke the existing RAG chain by passing the user's question into it.
            final_answer = (
                st.session_state.rag_chain.invoke(
                    standalone_question
                )
            )

        # Convert the retrieved LangChain Document chunks into simple dictionaries.
        saved_sources = []
        for document, distance in retrieval_results:
            relevance_score = round(
                (1 / (1+distance)) * 100,
                2
            )
            saved_sources.append(
                {
                    "content": document.page_content,
                    "metadata": document.metadata,
                    "score": relevance_score
                }
            )

        # Add the user's question to the saved chat history.
        st.session_state.messages.append(
            # Store the question as a user chat message.
            {"role": "user", "content": question}
        )

        # Add the AI answer to the saved chat history.
        st.session_state.messages.append(
            # Store the answer as an assistant chat message with its source chunks.
            {"role": "assistant", "content": final_answer, "sources": saved_sources}
        )

        # Rerun the Streamlit app so the new chat messages appear in the conversation area.
        st.rerun()
    # Run this block if the Ask button was clicked but the question box is empty.
    else:
        # Show a warning asking the user to enter a question first.
        st.warning("Please enter a question before clicking Ask.")
