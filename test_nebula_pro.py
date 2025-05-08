import os

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from data_preprocessing import parse_pdf, convert_to_documents, convert_img_to_tables

ROOT_DIR = "/root/autodl-tmp/multimodal-rag-finance/data"
DATA_DIR = os.path.join(ROOT_DIR, "img")
INPUT_DIR = os.path.join(DATA_DIR, "pdf-inputs")

raw_files = os.listdir(ROOT_DIR)
pdf_files = [file for file in raw_files if file.endswith(".pdf")]

embed_path = "/root/autodl-tmp/bge-m3"
embed_model = HuggingFaceEmbedding(embed_path)
llm = Ollama(model="qwen3:latest", request_timeout=3600.0)
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 50


vec_store = SimpleVectorStore()
graph_store = NebulaPropertyGraphStore(
    space="llamaindex", overwrite=True, refresh_schema=True  # 强制检查/刷新 schema
)

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

    index = PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=graph_store,
        vector_store=vec_store,
        show_progress=True,
    )

    index.storage_context.vector_store.persist("./data/vec/nebula_vec_store.json")
