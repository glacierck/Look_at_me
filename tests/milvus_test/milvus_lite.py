""" example.py based from pymilvus
"""
from __future__ import annotations

import pprint


import numpy as np
from milvus import default_server

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)
__all__ = ['Milvus']

# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. insert entities
#   4. create index
#   5. search
# 想象成pandas的DataFrame，列名是字段名，每一行是一个实体，实体由不同的字段组成
#################################################################################
# 2. field view
# +-+------------+------------+------------------+------------------------------+
# | | "pk"      | "random"   |    "embeddings"   |
# +-+------------+------------+------------------+------------------------------+
# |1|  VarChar  |    Double  |    FloatVector    |
# +-+------------+------------+------------------+------------------------------+
# |2|  VarChar  |    Double  |    FloatVector    |
# +-+------------+------------+------------------+------------------------------+
# |3|| VarChar  |    Double  |    FloatVector    |
# +-+------------+------------+------------------+------------------------------+
##############################################################################
# Data type of the data to insert must **match the schema of the collection**
# otherwise Milvus will raise exception.
class Milvus:
    def __init__(self, base_dir: str = 'test_milvus', **kwargs: dict):
        # 设置默认的Milvus服务器的存储位置。如果不设置，则默认会存储到 %APPDATA%/milvus-io/milvus-server 路径下
        self._server_start(base_dir)
        self._base_config_set(**kwargs)
        self.collection = self._create_collection()
        # self.collection.set_properties(properties={"collection.ttl.seconds": 1800})
        self.list_collections()
        print(f"\nMlilvus init done.")

    def _base_config_set(self, **kwargs: dict):

        # 定义常量名，包括集合名和字段名
        self._collection_name = kwargs.get('collection_name', 'Faces')
        # fields params
        self._id_field_param = {'name': 'id',
                                'dtype': DataType.INT64, 'description': "face_ids", 'is_primary': True}
        self._name_field_param = {'name': 'name',
                                  'dtype': DataType.VARCHAR, 'description': "names of faces", 'max_length': 50}
        self._embedding_field_param = {'name': 'normed_embedding',
                                       'dtype': DataType.FLOAT_VECTOR, 'description': "face_normed_embedding vector",
                                       'dim': 512}

        self._index_file_size = kwargs.get('index_file_size', 32)  # 存储索引的最大文件大小

        # 定义索引的参数
        self._index_type = ('IVF_FLAT', 'FLAT', 'IVF_SQ8', 'IVF_SQ8H', 'IVF_PQ', 'HNSW', 'ANNOY')
        self._metric_type = ('L2', 'IP')  # 度量类型 L2->欧式距离,不支持余弦相似度，IP->内积
        self._nlist = kwargs.get('nlist', 1024)
        self._nprobe = kwargs.get('nprobe', 256 // 13)

        # 搜索参数预备
        self._prepared_search_param = {
            "metric_type": self._metric_type[1],
            "params": {"nlist": self._nlist, 'nprobe': self._nprobe}}
        self._index_param = {
            "index_type": self._index_type[0],
            **self._prepared_search_param}
        # 指定搜素时返回的最大结果数
        self._top_k = kwargs.get('top_k', 1)
        # 指定集合搜素字段
        # expr: 指定筛选条件
        # partition_names: 指定分区名
        # output_fields: 指定额外的返回的字段
        # _async，_callback异步编程相关
        self._collection_search_param = {
            "param": self._prepared_search_param,
            "anns_field": self._embedding_field_param['name'],
            "limit": self._top_k,
            "expr": "id >= 0",
            "output_fields": [self._id_field_param['name'],
                              self._name_field_param['name']]}  # search doesn't support vector field as output_fields

    @property
    def milvus_params(self) -> dict:
        params = {
            'collection_name': self._collection_name,
            'collection_description': self.collection.description,
            'collection_schema': self.collection.schema,
            'collection_num_entities': self.collection.num_entities,
            'collection_primary_field': self.collection.primary_field,
            'collection_fields': self.collection.schema.fields,
            'index_file_size': self._index_file_size,
            'search_param': self._collection_search_param,
            'index_param': self._index_param,
        }
        return params

    @staticmethod
    def _server_start(base_dir: str = 'test_milvus'):
        default_server.set_base_dir(base_dir)
        default_server.cleanup()
        default_server.start()
        print(f"Milvus server is running on {default_server.server_address}")
        print(f"\nCreate connection...")
        connections.connect(host='127.0.0.1', port=default_server.listen_port)
        print(f"\nList connections:")
        print(connections.list_connections())

    # 创建一个的集合
    def _create_collection(self) -> Collection:
        # 如果集合已存在，则删除集合
        if utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)
        id_field = FieldSchema(**self._id_field_param)
        name_field = FieldSchema(**self._name_field_param)
        embedding_field = FieldSchema(**self._embedding_field_param)
        field = [id_field, name_field, embedding_field]
        # 允许了动态字段，即可以在插入数据时动态添加字段
        schema = CollectionSchema(fields=field,
                                  description="collection faces_info_collection")

        collection = Collection(name=self._collection_name, schema=schema,
                                properties={"collection.ttl.seconds": 1800})
        print("\ncollection created:", self._collection_name)
        return collection

    def insert(self, entities: list):
        print("\nInsert data...")
        self.collection.insert(entities)
        self.collection.flush()
        print("Done inserting data.")
        print(self.get_entity_num)
        self._create_index()
        utility.wait_for_index_building_complete(self._collection_name)
        # 将collection 加载到到内存中
        self.collection.load()
        # Check the loading progress and loading status
        print(utility.load_state(self._collection_name))
        # Output: <LoadState: Loaded>
        print(utility.loading_progress(self._collection_name))
        # Output: {'loading_progress': 100%}

    # 向集合中插入实体
    def insert_from_files(self, file_paths: list):  ### failed
        print("\nInsert data...")
        # 3. insert entities
        task_id = utility.do_bulk_insert(collection_name=self._collection_name,
                                         partition_name=self.collection.partitions[0].name, files=file_paths)
        task = utility.get_bulk_insert_state(task_id=task_id)
        print("Task state:", task.state_name)
        print("Imported files:", task.files)
        print("Collection name:", task.collection_name)
        print("Start time:", task.create_time_str)
        print("Entities ID array generated by this task:", task.ids)
        while task.state_name != 'Completed':
            task = utility.get_bulk_insert_state(task_id=task_id)
            print("Task state:", task.state_name)
            print("Imported row count:", task.row_count)
            if task.state == utility.BulkInsertState.ImportFailed:
                print("Failed reason:", task.failed_reason)
                raise Exception(task.failed_reason)
        self.collection.flush()
        print(self.get_entity_num)
        print("Done inserting data.")
        self._create_index()
        utility.wait_for_index_building_complete(self._collection_name)

    # 列出Milvus中的所有集合
    @staticmethod
    def list_collections():
        print("\nlist collections:")
        print(utility.list_collections())

    # 获取集合中的实体数量
    @property
    def get_entity_num(self):
        return self.collection.num_entities

    # 创建索引
    def _create_index(self):
        self.collection.create_index(field_name=self._embedding_field_param['name'],
                                     index_params=self._index_param)
        print("\nCreated index:\n{}".format(self.collection.index().params))

    # 搜索集合
    # Question: 是否可以进行异步搜索？
    # noinspection PyTypeChecker
    def search(self, search_vectors: list[np.ndarray]) -> list[list[dict]]:
        # search_vectors可以是多个向量
        print(f"\nSearching ...")
        results = self.collection.search(data=search_vectors, **self._collection_search_param)
        print("collecting results ...")
        ret_results = [[] for _ in range(len(results))]
        for i, hits in enumerate(results):
            for hit in hits:
                ret_results[i].append({
                    'score': hit.score,
                    "id": hit.entity.get(self._id_field_param['name']),
                    "name": hit.entity.get(self._name_field_param['name'])
                })
        pprint.pprint(f"Search results : {ret_results}")
        return ret_results

    # 删除集合中的所有实体,并且关闭服务器
    # question: 可以不删除吗？下次直接读取上一次的内容？
    def shut_down(self):
        # 释放内存
        self.collection.release()
        print(f"\nReleased collection : {self._collection_name} successfully !")
        self.collection.drop_index()
        print(f"Drop index: {self._collection_name} successfully !")
        self.collection.drop()
        print(f"Drop collection: {self._collection_name} successfully !")
        default_server.stop()
        print(f"Stop Milvus server successfully !")

    def has_collection(self):
        return utility.has_collection(self._collection_name)

    def __bool__(self):
        return self.get_entity_num > 0


def main():
    # 创建连接
    Faces_server = Milvus()


if __name__ == '__main__':
    main()
