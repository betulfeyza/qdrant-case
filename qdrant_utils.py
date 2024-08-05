import pyarrow.parquet as pq
from qdrant_client import models

def setup_qdrant_collection(client, collection_name):
    # Check if the collection exists
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]

    if collection_name in collection_names:
        print(f"Collection '{collection_name}' already exists.")
        client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted.")

    # Create the collection
    client.create_collection(
        collection_name=collection_name,
        # vectors_config=client.get_fastembed_vector_params()

        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )
    print(f"Collection '{collection_name}' created.")

def parquet_batch_generator(file_path, batch_size=128):
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()

def store_embeddings(client, collection_name, texts, labels, model):
    vectors = {
        'all-MiniLM-L6-v2': [
            arr.tolist()
            for arr in model.encode(
                sentences=texts,
                batch_size=128,
                normalize_embeddings=True,
            )
        ]
    }
    
    points = [
        models.PointStruct(
            id=index,
            vector=vectors['all-MiniLM-L6-v2'][index],
            payload={"text": text, "label": label}
        )
        for index, (text, label) in enumerate(zip(texts, labels))
    ]
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )


# # client.upload_collection(
# #         collection_name="vector_store",
# #         vectors=[
# #             {client.get_vector_field_name(): arr.tolist()}
# #             for arr in SentenceTransformer(constants.MODEL_NAME).encode(
# #                 sentences=docs,  # piece of data on each iteration
# #                 batch_size=256,
# #                 normalize_embeddings=True,
# #             )
# #         ],
# #         payload=metadata,
# #         ids=ids,
# #         wait=True,
# #         batch_size=256,
# #     )


 