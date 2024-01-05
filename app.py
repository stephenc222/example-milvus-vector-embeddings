from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from embedding_util import generate_embeddings


def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise


def create_collection(name, fields, description, consistency_level="Strong"):
    schema = CollectionSchema(fields, description)
    collection = Collection(name, schema, consistency_level=consistency_level)
    return collection


def insert_data(collection, entities):
    insert_result = collection.insert(entities)
    collection.flush()
    print(
        f"Inserted data into '{collection.name}'. Number of entities: {collection.num_entities}")
    return insert_result


def create_index(collection, field_name, index_type, metric_type, params):
    index = {"index_type": index_type,
             "metric_type": metric_type, "params": params}
    collection.create_index(field_name, index)
    print(f"Index '{index_type}' created for field '{field_name}'.")


def search_and_query(collection, search_vectors, search_field, search_params):
    collection.load()

    # Vector search
    result = collection.search(
        search_vectors, search_field, search_params, limit=3, output_fields=["source"])
    print_search_results(result, "Vector search results:")


def print_search_results(results, message):
    print(message)
    for hits in results:
        for hit in hits:
            print(f"Hit: {hit}, source field: {hit.entity.get('source')}")


def delete_entities(collection, expr):
    collection.delete(expr)
    print(f"Deleted entities where {expr}")


def drop_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Dropped collection '{collection_name}'.")


# Main
dim = 768  # Adjust the dimension as per your model's output

connect_to_milvus()

fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR,
                is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]
collection = create_collection(
    "hello_milvus", fields, "Collection for demo purposes")

documents = [
    "A group of vibrant parrots chatter loudly, sharing stories of their tropical adventures.",
    "The mathematician found solace in numbers, deciphering the hidden patterns of the universe.",
    "The robot, with its intricate circuitry and precise movements, assembles the devices swiftly.",
    "The chef, with a sprinkle of spices and a dash of love, creates culinary masterpieces.",
    "The ancient tree, with its gnarled branches and deep roots, whispers secrets of the past.",
    "The detective, with keen observation and logical reasoning, unravels the intricate web of clues.",
    "The sunset paints the sky with shades of orange, pink, and purple, reflecting on the calm sea.",
    "In the dense forest, the howl of a lone wolf echoes, blending with the symphony of the night.",
    "The dancer, with graceful moves and expressive gestures, tells a story without uttering a word.",
    "In the quantum realm, particles flicker in and out of existence, dancing to the tunes of probability."
]


embeddings = [generate_embeddings(doc) for doc in documents]
entities = [
    [str(i) for i in range(len(documents))],
    [str(doc) for doc in documents],
    embeddings
]
insert_result = insert_data(collection, entities)

create_index(collection, "embeddings", "IVF_FLAT", "L2", {"nlist": 128})

query = "Give me some content about the ocean"
query_vector = generate_embeddings(query)
search_and_query(collection, [query_vector], "embeddings", {
                 "metric_type": "L2", "params": {"nprobe": 10}})

delete_entities(
    collection, f'pk in ["{insert_result.primary_keys[0]}" , "{insert_result.primary_keys[1]}"]')
drop_collection("hello_milvus")
