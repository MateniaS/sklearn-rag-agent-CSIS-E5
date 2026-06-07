from langchain_core.tools import tool

from react_agent import build_context, retrieve_chunks


VALID_TOPIC_FILTERS = [
    "general_intro",
    "preprocessing",
    "pipelines",
    "train_test_split",
    "cross_validation",
    "hyperparameter_tuning",
    "metrics",
    "logistic_regression",
    "random_forest",
    "random_forest_classifier",
]

_last_retrieval = {"points": []}


def get_last_retrieval():
    return _last_retrieval["points"]


def reset_last_retrieval():
    _last_retrieval["points"] = []


def create_retrieval_tools(qdrant_client, openai_client):
    @tool
    def rag_retriever(question: str, top_k: int = 5) -> str:
        """Search the full scikit-learn documentation corpus using semantic retrieval.

        Use this for broad or multi-step workflow questions that may span several topics.
        """
        points = retrieve_chunks(
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            question=question,
            top_k=top_k,
        )
        _last_retrieval["points"] = points
        return build_context(points)

    @tool
    def metadata_filtered_retriever(
        question: str,
        topic_filter: str,
        top_k: int = 5,
    ) -> str:
        """Search scikit-learn documentation with metadata filtering on topic.

        Use this when the question targets a specific topic or API page.
        Valid topic_filter values:
        general_intro, preprocessing, pipelines, train_test_split,
        cross_validation, hyperparameter_tuning, metrics, logistic_regression,
        random_forest, random_forest_classifier
        """
        points = retrieve_chunks(
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            question=question,
            top_k=top_k,
            topic_filter=topic_filter,
        )
        _last_retrieval["points"] = points
        return build_context(points)

    return [rag_retriever, metadata_filtered_retriever]
