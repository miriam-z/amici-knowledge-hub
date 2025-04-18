import os
import pandas as pd
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

# Connect to MongoDB Atlas
client = MongoClient(os.getenv("MONGO_URI"))
db = client["test_qa"]
collection = db["stable_diffusion_qna"]

# Clean start: delete all existing documents but preserve the index
collection.delete_many({})
print("Existing documents removed, preserving index.")

# Create embedding function
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Prepare all documents
all_documents = []

# Define data folder
data_folder = "data"

# Iterate over all CSV files in data folder
for filename in os.listdir(data_folder):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(data_folder, filename)
    df = pd.read_csv(file_path)

    print(f"Processing file: {filename}")

    # Detect type based on file name or columns
    if "question" in df.columns and "answer" in df.columns:
        file_type = "qa"
    elif "content" in df.columns:
        file_type = "chunk"
    else:
        print(f"Skipped unrecognized format: {filename}")
        continue

    # Process Q&A format
    if file_type == "qa":
        documents = [
            Document(
                page_content=f"Q: {row['question']}\nA: {row['answer']}",
                metadata={
                    "source": row.get("source", ""),
                    "question": row.get("question", ""),
                    "source_type": "faq",
                },
            )
            for _, row in df.iterrows()
        ]
    # Process chunked format
    elif file_type == "chunk":
        documents = [
            Document(
                page_content=row["content"],
                metadata={
                    "source": row.get("source", ""),
                    "title": row.get("title", ""),
                    "chunk_id": int(row.get("chunk_id", -1)),
                    "source_type": row.get("source_type", ""),
                },
            )
            for _, row in df.iterrows()
        ]
    else:
        documents = []

    print(f"Processed {len(documents)} documents from {filename}")
    all_documents.extend(documents)

print(f"Total documents to upload: {len(all_documents)}")

# Insert all documents into MongoDB Atlas Vector Search
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    documents=all_documents,
    embedding=embeddings,
    collection=collection,
    index_name="default",  # make sure this matches your Atlas index
)

print("Data added successfully into MongoDB Atlas Vector Search.")
