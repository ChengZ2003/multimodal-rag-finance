# from ragas.testset.generator import TestDataset
# from ragas.testset.evolutions import DataRow
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
from llama_index.core import Document
from typing import Optional
import pandas as pd
import os
import json
from tqdm import tqdm


def parse_pdf(
    pdf,
    extract_image_block_output_dir: Optional[str] = None,
    extract_images_in_pdf: bool = False,
):
    """
    Parse a PDF file using the `unstructured` library.
    """

    assert os.path.exists(pdf), f"PDF file {pdf} does not exist"
    if extract_images_in_pdf:
        return partition_pdf(
            pdf,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=extract_image_block_output_dir,
        )
    else:
        return partition_pdf(pdf, strategy="hi_res", extract_images_in_pdf=False)


def convert_img_to_tables(raw_docs, output_dir):
    """
    Convert the image blocks to tables using the QwenVL model.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    from qwen_convertor import QwenConvertorFactory

    factory = QwenConvertorFactory()
    qwen_convertor = factory.get_instance()
    documents = []
    img_captions = []

    for doc in tqdm(raw_docs, "Converting images to tables"):
        if doc.to_dict()["type"] in ["Image", "Table"]:
            img_path = doc.to_dict()["metadata"]["image_path"]
            converted_text = qwen_convertor.convert(img_path=img_path)
            img_captions.append(
                {"image_path": img_path.split("/")[-1], "caption": converted_text}
            )
            converted_doc = doc
            converted_doc.text = converted_text
            documents.append(converted_doc)
        else:
            documents.append(doc)

    img_captions_df = pd.DataFrame(img_captions)
    img_captions_df.to_csv(f"{output_dir}/img_captions.csv", index=False)

    qwen_convertor.clear_GPU_mem()

    return documents


def convert_to_documents(documents, max_characters=512, overlap=50):
    """
    convert the partitioned documents to llamaindex Document objects.
    """

    chunks = chunk_elements(documents, max_characters=max_characters, overlap=overlap)
    documents = []
    text_seq = []
    for chunk in tqdm(chunks, desc="Converting to documents"):
        text = chunk.to_dict()["text"]
        text_seq.append(text)
        document = Document(
            doc_id=chunk.to_dict()["element_id"],
            text=text,
            metadata={
                "page_number": chunk.to_dict()["metadata"]["page_number"],
                "filename": chunk.to_dict()["metadata"]["filename"],
            },
        )
        documents.append(document)

    return documents, text_seq


# def load_dataset(dataset_path: str) -> TestDataset:
#     """
#     Load the dataset from a CSV file.
#     """

#     assert dataset_path.endswith('.csv'), 'Dataset file must be a CSV file.'

#     df = pd.read_csv(dataset_path,
#                      quotechar='"',
#                      skipinitialspace=True,)

#     data_rows = []
#     for _, row in df.iterrows():
#         data_row = DataRow(
#             question=row['question'],
#             contexts=eval(row['contexts']),
#             ground_truth=row['ground_truth'],
#             evolution_type=row['evolution_type'],
#             metadata=eval(row['metadata'])
#         )
#         data_rows.append(data_row)

#     test_dataset = TestDataset(test_data=data_rows).to_dataset().to_dict()
#     return test_dataset


def load_img_captions(raw_docs, csv_path):
    """
    Load the image captions from a CSV file.
    """

    img_caption = pd.read_csv(csv_path)
    pdf_id = csv_path.split("/")[-2]
    img_ids = []
    documents = []

    for full_img_path in img_caption["image_path"]:
        img_id = full_img_path.split("/")[-1]
        img_ids.append(img_id)

    for doc in raw_docs:
        if doc.to_dict()["type"] in ["Table", "Image"]:
            doc_img_id = doc.to_dict()["metadata"]["image_path"].split("/")[-1]
            if doc_img_id in img_ids:
                caption = img_caption.loc[
                    img_caption["image_path"] == f"./images/{pdf_id}/{doc_img_id}",
                    "caption",
                ].values[0]
                converted_doc = doc
                converted_doc.text = caption
                documents.append(converted_doc)
        else:
            documents.append(doc)

    return documents


def ragas2beir(csv_file_path, output_dir="beir_dataset"):
    """
    Convert the RAGAS dataset to BEIR format.
    """

    assert os.path.exists(csv_file_path), f"CSV file {csv_file_path} does not exist"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = pd.read_csv(csv_file_path)
    corpus = []

    for idx, row in data.iterrows():
        contexts = eval(row["contexts"])
        for context in contexts:
            corpus.append(
                {
                    "_id": f"{idx}",
                    "text": context,
                }
            )

    corpus_file_path = f"{output_dir}/corpus.jsonl"
    with open(corpus_file_path, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")

    queries = []
    for idx, row in data.iterrows():
        queries.append({"_id": f"{idx}", "text": row["question"]})

    queries_file_path = f"{output_dir}/queries.jsonl"
    with open(queries_file_path, "w") as f:
        for query in queries:
            f.write(json.dumps(query) + "\n")

    qrels = []
    for idx, row in data.iterrows():
        contexts = eval(row["contexts"])
        for context_idx, context in enumerate(contexts):
            qrels.append(f"{idx}\t{idx}\t1")

    qrels_file_path = f"{output_dir}/qrels.tsv"
    with open(qrels_file_path, "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qrel in qrels:
            f.write(qrel + "\n")
