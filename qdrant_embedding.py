# from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_utils import setup_qdrant_collection, parquet_batch_generator, store_embeddings

def process_qdrant_embeddings(client, collection_name):
    # Define the parameters for the Qdrant client
    # host = 'localhost'
    # port = 6333

    # Create the Qdrant client
    # client = QdrantClient(url=f"http://{host}:{port}")

    # Define the collection name
    # collection_name = "my_collection"

    # Setup Qdrant collection
    setup_qdrant_collection(client, collection_name)

    # Load the Sentence Transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Process and store embeddings
    file_path = 'preprocessed_poem_sentiment_train.parquet'

    for i, batch in enumerate(parquet_batch_generator(file_path, batch_size=128)):
        texts = batch['verse_text'].tolist()  
        labels = batch['label'].tolist() 
        store_embeddings(client, collection_name, texts, labels, model)
        print(f"Batch {i+1} upserted.")

    print("All batches upserted.")

if __name__ == "__main__":
    process_qdrant_embeddings()