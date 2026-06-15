"""Load a PDF, split it into chunks, create embeddings, and save them in FAISS."""

# Import Path so we can work with file and folder paths in a clean way.
from pathlib import Path

# Import PyPDFLoader from LangChain to read PDF files page by page.
from langchain_community.document_loaders import PyPDFLoader

# Import FAISS so we can store and search document embeddings locally.
from langchain_community.vectorstores import FAISS

# Import HuggingFaceEmbeddings to convert text chunks into numeric vectors.
from langchain_huggingface import HuggingFaceEmbeddings

# Import RecursiveCharacterTextSplitter to break large page text into chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Store the path to the folder where PDF documents will be placed.
DATA_FOLDER = Path("data")

# Store the path to the folder where the FAISS database will be saved.
VECTORSTORE_FOLDER = Path("vectorstore")

# Create the vectorstore folder if it does not already exist.
VECTORSTORE_FOLDER.mkdir(exist_ok=True)

# Find every PDF file inside the data folder.
pdf_files = list(DATA_FOLDER.glob("*.pdf"))

# Stop the program with a helpful message if no PDF file is found.
if not pdf_files:
    raise FileNotFoundError("No PDF files found in the data folder.")

# Select the first PDF file found in the data folder.
pdf_path = pdf_files[0]

# Create a LangChain PDF loader for the selected PDF file.
loader = PyPDFLoader(str(pdf_path))

# Load the PDF into a list of LangChain Document objects, one per page.
pages = loader.load()

# Create a text splitter that tries to split text at natural boundaries.
text_splitter = RecursiveCharacterTextSplitter(
    # Set each chunk to contain about 500 characters.
    chunk_size=500,
    # Keep 50 characters of overlap between chunks for better context.
    chunk_overlap=50,
)

# Split the loaded PDF pages into smaller LangChain Document chunks.
chunks = text_splitter.split_documents(pages)

# Create an embeddings model using a small sentence-transformers model.
embeddings = HuggingFaceEmbeddings(
    # Choose the exact HuggingFace model that will turn text into vectors.
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Get only the text content from each chunk, because embeddings work on text.
chunk_texts = [chunk.page_content for chunk in chunks]

# Convert every chunk of text into an embedding vector.
chunk_embeddings = embeddings.embed_documents(chunk_texts)

# Get the metadata from each chunk, such as the source file and page number.
chunk_metadatas = [chunk.metadata for chunk in chunks]

# Pair each chunk of text with its matching embedding vector.
text_embedding_pairs = list(zip(chunk_texts, chunk_embeddings))

# Create a FAISS vector database from the chunk texts, embeddings, and metadata.
vectorstore = FAISS.from_embeddings(
    # Give FAISS each chunk of text together with its already-created embedding.
    text_embeddings=text_embedding_pairs,
    # Give FAISS the same embedding model so it can embed future user questions.
    embedding=embeddings,
    # Store each chunk's metadata alongside the chunk text and vector.
    metadatas=chunk_metadatas,
)

# Save the FAISS vector database files inside the vectorstore folder.
vectorstore.save_local(str(VECTORSTORE_FOLDER))

# Print the PDF filename so the user knows which file was loaded.
print(f"Loaded PDF: {pdf_path.name}")

# Print the number of pages loaded from the PDF.
print(f"Number of pages loaded: {len(pages)}")

# Print the total number of chunks created from the PDF pages.
print(f"Total number of chunks created: {len(chunks)}")

# Print how many chunks successfully received embeddings.
print(f"Number of chunks that received embeddings: {len(chunk_embeddings)}")

# Print a success message so the user knows the FAISS database was saved.
print(f"FAISS vector database saved successfully in: {VECTORSTORE_FOLDER}")
