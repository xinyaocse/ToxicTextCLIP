from typing import List, Dict, Union
from pymilvus import MilvusClient, DataType
import os
import shutil


TEXT_TABLE = "textsearch"
IMAGE_TABLE = "imagesearch"
def batch(iterable, n = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_database(database_path:str, database_name = None, create_new:bool = False):
    '''
    Args:
        database_path: Database storage files path
        create_new: Do you want to create a new dataset? Setting it to True will delete the current existing database
        database_name: Database Name
    Returns:
    '''
    if create_new:
        # Recursive deletion of folders includes the folder itself
        shutil.rmtree(database_path, ignore_errors=True)
    os.makedirs(database_path, exist_ok=True)
    if database_name and not database_name.endswith('db'):
        database_name = database_name + '.db'
    DB_PATH = os.path.join(database_path, database_name if database_name else "milvus.db")
    return MilvusClient(uri=DB_PATH)


def create_tables(client, text_dim=1024, image_dim=1024, create_new:bool = False):
    if create_new:
        if client.has_collection(collection_name=TEXT_TABLE):
            client.drop_collection(collection_name=TEXT_TABLE)
        if client.has_collection(collection_name=IMAGE_TABLE):
            client.drop_collection(collection_name=IMAGE_TABLE)

    text_schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    text_schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id = True)  # Primary key ID
    text_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=text_dim)  # Stored vectors
    text_schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=128)  # plaintext
    text_schema.add_field(field_name="class", datatype=DataType.VARCHAR,
                          max_length=32)  # Category, such as clean, test1, etc., can be whether it is clean or not, or it can be an experimental code
    text_schema.add_field(field_name="extra_data", datatype=DataType.VARCHAR, max_length=256, default_value = "")  # Additional information, reserved fields, generally recommended to convert JSON to string format
    client.create_collection(
        collection_name=TEXT_TABLE,
        schema=text_schema,
    )

    text_index_params = MilvusClient.prepare_index_params()
    text_index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="FLAT",
        index_name="text_vector_index",
    )
    client.create_index(
        collection_name=TEXT_TABLE,
        index_params=text_index_params,
        sync=False  # Whether to wait for index creation to complete before returning. Defaults to True.
    )

    iamge_schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    iamge_schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True,auto_id = True)  # Primary key ID
    iamge_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=image_dim)  # Stored vectors
    iamge_schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=128)  # plaintext
    iamge_schema.add_field(field_name="class", datatype=DataType.VARCHAR,
                           max_length=32)  
    iamge_schema.add_field(field_name="extra_data", datatype=DataType.VARCHAR, max_length=256, default_value = "")  
    client.create_collection(
        collection_name=IMAGE_TABLE,
        schema=iamge_schema,
    )

    image_index_params = MilvusClient.prepare_index_params()
    image_index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="FLAT",
        index_name="image_vector_index",
    )
    client.create_index(
        collection_name=IMAGE_TABLE,
        index_params=image_index_params,
        sync=False  # Whether to wait for index creation to complete before returning. Defaults to True.
    )

def create_data(is_text:bool, embedding:List[List[float]], batch_data:Union[List[List[str]],List[str]], class_type:Union[str,List[str]], extra_data:Union[str,List[str]] = "") -> List[Dict[str,Union[List[float],str,]]]:
    # Initialization result list
    result = []

    if isinstance(class_type,str):
        # Traverse embeddings and batch_data
        for emb, data in zip(embedding, batch_data):
            if is_text:
                result.append({
                    "vector": emb,
                    "text": data,
                    "class": class_type,
                    "extra_data": extra_data
                })
            else:
                result.append({
                    "vector": emb,
                    "image_path": data,
                    "class": class_type,
                    "extra_data": extra_data
                })
    else:
        if isinstance(extra_data,str):
            extra_data = [extra_data] * len(embedding)
        for emb, data,cls,extdata in zip(embedding, batch_data,class_type,extra_data):
            if is_text:
                result.append({
                    "vector": emb,
                    "text": data,
                    "class": cls,
                    "extra_data": extdata
                })
            else:
                result.append({
                    "vector": emb,
                    "image_path": data,
                    "class": cls,
                    "extra_data": extdata
                })


    return result



