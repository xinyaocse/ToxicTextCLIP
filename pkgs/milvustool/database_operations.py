from typing import List, Dict, Union
from pymilvus import MilvusClient, DataType
import os
import shutil


TEXT_TABLE = "textsearch"
IMAGE_TABLE = "imagesearch"


def get_database(database_path:str, database_name = None, create_new:bool = False):
    '''
    Args:
        database_path: Database storage directory
        create_new: Setting it to True will delete the existing database
        database_name: database name
    Returns:
    '''
    if create_new:
        #Recursive deletion of folders includes the folder itself
        shutil.rmtree(database_path, ignore_errors=True)
    os.makedirs(database_path, exist_ok=True)
    if database_name and not database_name.endswith('db'):
        database_name = database_name + '.db'
    DB_PATH = os.path.join(database_path, database_name if database_name else "milvus.db")
    return MilvusClient(uri=DB_PATH)


def create_tables(client, text_dim=1024, image_dim=1024, create_new:bool = False):
    '''
    Create two collections, textsearch and imagesearch, and index vector fields.
    '''
    if create_new:
        for name in [TEXT_TABLE, IMAGE_TABLE]:
            try:
                if client.has_collection(collection_name=name):
                    client.drop_collection(collection_name=name)
            except Exception:
                # It may not exist during the first run, ignore it
                pass
    if not client.has_collection(collection_name=TEXT_TABLE):

        text_schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )
        text_schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id = True)  # Primary key ID
        text_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=text_dim)  # Stored vectors
        text_schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=128)  # plaintext
        text_schema.add_field(field_name="class", datatype=DataType.VARCHAR,
                          max_length=32)  # Category, such as clean, test1, etc., can be whether it is clean or not, or it can be an experimental code
        text_schema.add_field(field_name="extra_data", datatype=DataType.VARCHAR, max_length=512)  # Additional information, reserved fields, generally recommended to convert JSON to string format

        text_index_params = client.prepare_index_params()
        text_index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="AUTOINDEX",
            index_name="text_vector_index",
        )

        client.create_collection(
            collection_name=TEXT_TABLE,
            schema=text_schema,
            index_params=text_index_params,
        )

    if not client.has_collection(collection_name=IMAGE_TABLE):

        image_schema = client.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )
        image_schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True,auto_id = True)  # Primary key ID
        image_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=image_dim)  # Stored vectors
        image_schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=256)  # plaintext
        image_schema.add_field(field_name="class", datatype=DataType.VARCHAR,
                               max_length=32)  # Category, such as clean, test1, etc., can be whether it is clean or not, or it can be an experimental code
        image_schema.add_field(field_name="extra_data", datatype=DataType.VARCHAR, max_length=512)  # Additional information, reserved fields, generally recommended to convert JSON to string format

        image_index_params = client.prepare_index_params()
        image_index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="AUTOINDEX",
            index_name="image_vector_index",
        )

        client.create_collection(
            collection_name=IMAGE_TABLE,
            schema=image_schema,
            index_params=image_index_params,
        )


def create_data(is_text:bool, embedding:List[List[float]], batch_data:List[str], class_type:Union[str,List[str]], extra_data:Union[str,List[str]] = "") -> List[Dict[str,Union[List[float],str,]]]:
    assert len(embedding) == len(batch_data), "The length of embedding and batch_data is inconsistent"
    # process class_type
    if isinstance(class_type, str):
        class_list = [class_type] * len(embedding)
    else:
        assert len(class_type) == len(embedding), "class_type length is inconsistent with embedding"
        class_list = class_type

    # process extra_data
    if isinstance(extra_data, str):
        extra_list = [extra_data] * len(embedding)
    else:
        assert len(extra_data) == len(embedding), "The length of extra_data is inconsistent with the embedding"
        extra_list = extra_data
    # Initialization result list
    result = []

    for emb, data, cls, ext in zip(embedding, batch_data, class_list, extra_list):
        base = {
            "vector": emb,
            "class": cls,
            "extra_data": ext,
        }
        if is_text:
            base["text"] = data
        else:
            base["image_path"] = data
        result.append(base)

    return result
