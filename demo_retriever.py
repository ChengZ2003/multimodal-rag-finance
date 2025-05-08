from llama_index.core.retrievers import (
    BaseRetriever,
    VectorContextRetriever,
    TextToCypherRetriever,
)
from llama_index.core.graph_stores import PropertyGraphStore
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core import QueryBundle
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)

from llama_index.core.schema import NodeWithScore

from typing import Optional, Any, Union, List


class MyCustomRetriever(BaseRetriever):
    """Custom retriever with cohere reranking."""

    def __init__(
        self,
        ## vector context retriever params
        vector_retriever,
        kg_retriever,
        ## text-to-cypher params
        # llm: Optional[LLM] = None,
        # text_to_cypher_template: Optional[Union[PromptTemplate, str]] = None,
        ## cohere reranker params
        # cohere_api_key: Optional[str] = None,
        # cohere_top_n: int = 2,
        # **kwargs: Any,
    ) -> None:
        """Uses any kwargs passed in from class constructor."""

        self.vector_retriever = vector_retriever
        self.kg_retriever = kg_retriever
        self.reranker = FlagEmbeddingReranker(model="/root/autodl-tmp/bge-reranker-v2-m3", top_n=5)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Define custom retriever with reranking.

        Could return `str`, `TextNode`, `NodeWithScore`, or a list of those.
        """
        nodes_1 = self.vector_retriever.retrieve(query_bundle)
        nodes_2 = self.kg_retriever.retrieve(query_bundle)
        reranked_nodes = self.reranker.postprocess_nodes(
            nodes_1 + nodes_2, query_bundle=query_bundle
        )

        ## TMP: please change
        # final_text = "\n\n".join(
        #     [n.get_content(metadata_mode="llm") for n in reranked_nodes]
        # )

        return reranked_nodes

