import os

from llama_index.core import Settings
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext
from llama_index.core import KnowledgeGraphIndex
from llama_index.graph_stores.nebula import NebulaGraphStore

from data_preprocessing import parse_pdf, convert_to_documents, convert_img_to_tables

embed_path = "/root/autodl-tmp/bge-m3"
embed_model = HuggingFaceEmbedding(embed_path)
llm = Ollama(model="qwen3:latest", request_timeout=3600.0)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 50

ROOT_DIR = "/root/autodl-tmp/multimodal-rag-finance/data"
DATA_DIR = os.path.join(ROOT_DIR, "img")
INPUT_DIR = os.path.join(DATA_DIR, "pdf-inputs")

raw_files = os.listdir(ROOT_DIR)
pdf_files = [file for file in raw_files if file.endswith(".pdf")]

os.environ['NEBULA_USER'] = 'root'
os.environ['NEBULA_PASSWORD'] = 'nebula'
os.environ['NEBULA_ADDRESS'] = '127.0.0.1:9669'

space_name = 'llamaindex'
edge_types, rel_prop_names = ['relationship'], ['relationship']
tags = ["entity"]

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

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

    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=10,
        include_embeddings=True,
        show_progress=True
    )