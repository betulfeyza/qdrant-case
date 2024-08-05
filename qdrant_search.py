from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
# from qdrant_client.http import models
from loguru import logger
from pydantic import BaseModel, Field

# Configure loguru
logger.add("app.log", rotation="500 MB")

class SearchParams(BaseModel):
    query: str
    collection_name: str
    limit: int = Field(default=10, ge=1)
    score_threshold: float = Field(default=0.5, ge=0, le=1)
    sentiment: Optional[str] = Field(default=None, pattern="^(positive|negative|no_impact|mixed)$")

def search_similar_points(
    client: QdrantClient,
    model: SentenceTransformer,
    params: SearchParams
) -> List[models.ScoredPoint]:
    """
    Search for similar points in the Qdrant vector store.

    Args:
        client: QdrantClient instance
        model: SentenceTransformer model
        params: SearchParams instance

    Returns:
        List of ScoredPoint objects
    """
    query_vector = model.encode(params.query).tolist()

    # search_params = models.SearchParams(
    #     hnsw_ef=128,
    #     exact=False
    # )

    query_filter = 'positive'
    if params.sentiment:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="label",
                    match=models.MatchValue(value=params.sentiment)
                )
            ]
        )

    results = client.search(
        collection_name=params.collection_name,
        query_filter=query_filter,
        query_vector=query_vector,
        limit=params.limit,
        score_threshold=params.score_threshold,
        # search_params=search_params
    )

    logger.info(f"Found {len(results)} similar points for query: {params.query}")
    return results

def recommend_points(
    client: QdrantClient,
    model: SentenceTransformer,
    params: SearchParams,
    positive: List[str],
    negative: List[str]
) -> List[models.ScoredPoint]:
    """
    Recommend points based on positive and negative examples.

    Args:
        client: QdrantClient instance
        model: SentenceTransformer model
        params: SearchParams instance
        positive: List of positive example texts
        negative: List of negative example texts

    Returns:
        List of ScoredPoint objects
    """
    logger.info(f"Starting recommendation with query: {params.query}")
    logger.info(f"Positive examples: {positive}")
    logger.info(f"Negative examples: {negative}")
    logger.info(f"Sentiment filter: {params.sentiment}")
    logger.info(f"Score threshold: {params.score_threshold}")
    logger.info(f"Collection name: {params.collection_name}")

    positive_vectors = [model.encode(text).tolist() for text in positive]
    negative_vectors = [model.encode(text).tolist() for text in negative]

    logger.info(f"Encoded {len(positive_vectors)} positive vectors and {len(negative_vectors)} negative vectors")

    query_filter = None
    if params.sentiment:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="label",
                    match=models.MatchValue(value=params.sentiment)
                )
            ]
        )

    results = client.recommend(
        collection_name=params.collection_name,
        positive=positive_vectors,
        negative=negative_vectors,
        query_filter=query_filter,
        limit=params.limit,
        score_threshold=params.score_threshold
    )

    logger.info(f"Found {len(results)} recommended points")
    return results


