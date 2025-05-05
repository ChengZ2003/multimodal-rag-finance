import os

from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from hybrid_dev import ExampleEmbeddingFunction
from data_preprocessing import parse_pdf, convert_to_documents, convert_img_to_tables

embed_model = HuggingFaceEmbedding(model_name="/root/autodl-tmp/bge-m3", device="cuda:0")
Settings.embed_model = embed_model

ROOT_DIR = "/root/autodl-tmp/multimodal-rag-finance/data"
DATA_DIR = os.path.join(ROOT_DIR, "img")
INPUT_DIR = os.path.join(DATA_DIR, "pdf-inputs")

raw_files = os.listdir(ROOT_DIR)
pdf_files = [file for file in raw_files if file.endswith(".pdf")]

for file in pdf_files:
    file_id = file.split(".")[0]
    file_data_path = os.path.join(DATA_DIR, file_id)
    raw_docs = parse_pdf(
        os.path.join(ROOT_DIR, file),
        extract_image_block_output_dir=os.path.join(
            file_data_path, "images"
        ),
        extract_images_in_pdf=True,
    )
    docs = convert_img_to_tables(raw_docs, file_data_path)
    documents, text_seq = convert_to_documents(docs)
    vector_store = MilvusVectorStore(
        uri="./milvus_demo.db",
        token="root:Milvus",
        collection_name=f"doc_{file_id}",
        dim=1024,
        overwrite=True,
        enable_sparse=True,
        sparse_embedding_function=ExampleEmbeddingFunction(),
        hybrid_ranker="RRFRanker",
        hybrid_ranker_params={"k": 60},
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )