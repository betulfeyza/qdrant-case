import argparse
from poem_sentiment import process_poem_sentiment
from qdrant_embedding import process_qdrant_embeddings
from qdrant_search import search_similar_points, recommend_points, SearchParams
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description="Run poem sentiment analysis, Qdrant embeddings, or search/recommend points")
    parser.add_argument('--run', choices=['sentiment', 'embeddings', 'search', 'recommend', 'all'], default='all',
                        help="Choose which process to run: 'sentiment', 'embeddings', 'search', 'recommend', or 'all' (default)")
    parser.add_argument('--query', type=str, help="Query text for search or recommend")
    parser.add_argument('--collection', type=str, default='my_collection', help="Collection name in Qdrant")
    parser.add_argument('--limit', type=int, default=10, help="Limit for search or recommend results")
    parser.add_argument('--score_threshold', type=float, default=0.5, help="Score threshold for search or recommend")
    parser.add_argument('--sentiment', type=str, choices=['positive', 'negative', 'no_impact', 'mixed'], help="Sentiment filter for search or recommend")
    parser.add_argument('--positive', nargs='+', help="Positive examples for recommend")
    parser.add_argument('--negative', nargs='+', help="Negative examples for recommend")
    args = parser.parse_args()

    client = QdrantClient(url="http://localhost:6333")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # client.set_model('sentence-transformers/all-MiniLM-L6-v2')

    if args.run in ['sentiment', 'all']:
        print("Running poem sentiment analysis...")
        process_poem_sentiment()
        print("Poem sentiment analysis complete.")

    if args.run in ['embeddings', 'all']:
        print("Running Qdrant embeddings process...")
        process_qdrant_embeddings(client, 'test')
        print("Qdrant embeddings process complete.")

    if args.run == 'search':
        if not args.query:
            print("Query text is required for search")
            return
        params = SearchParams(query=args.query, collection_name='test', limit=args.limit, score_threshold=args.score_threshold, sentiment=args.sentiment)
        results = search_similar_points(client, model, params)
        for result in results:
            print(result)

    if args.run == 'recommend':
        if not args.query or not args.positive or not args.negative:
            print("Positive and negative examples are required for recommend")
            return
        
        # Try to recommend with the provided score_threshold
        params = SearchParams(query=args.query, collection_name=args.collection, limit=args.limit, score_threshold=args.score_threshold, sentiment=args.sentiment)
        results = recommend_points(client, model, params, positive=args.positive, negative=args.negative)
        
        # Retry with lower score_threshold if no results found
        if len(results) == 0:
            print("No recommended points found with the initial score threshold. Retrying with a lower threshold.")
            for threshold in [0.4, 0.3, 0.2, 0.1, 0.0]:
                params.score_threshold = threshold
                results = recommend_points(client, model, params, positive=args.positive, negative=args.negative)
                if len(results) > 0:
                    print(f"Recommended points found with score threshold {threshold}")
                    break
        
        for result in results:
            print(result)

if __name__ == "__main__":
    main()



# python main.py --run search --query "love and nature" --sentiment positive --limit 5 --score_threshold 0.6

# python main.py --run recommend --query "exciting adventure" --positive "amazing trip" "wonderful journey" --negative "boring day" "dull moment" --collection my_collection --limit 5 --score_threshold 0.7
# python main.py --run recommend --query "rising said" --positive "amazing day" "wonderful tree" --negative "boring home" "tiring moment" --collection my_collection --limit 5 --score_threshold 0.7
