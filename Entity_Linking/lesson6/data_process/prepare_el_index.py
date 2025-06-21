
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

import hashlib

from transformers import BasicTokenizer


def up_index_for_icd_el(icd_data, es_conn, index_name, bulk_size=1000):

    bulk_samples = []
    count = 0
    for i, row in tqdm(icd_data.iterrows()):
        icd_code = row["icd_code"].strip()
        if len(icd_code < 7):
            continue
        entity_name = row["entity_name"]
        entity_name = basic_tokenizer._clean_text(entity_name.strip())
        if not entity_name:
            continue
        if not icd_code:
            continue

        m = hashlib.md5()
        m.update(f"{entity_name}--{icd_code}".encode("utf8"))
        entity_id = m.hexdigest()
        # print(entity_id)

        content = {
            "entity_name": entity_name,
            "entity_id": entity_id,
            "icd_code": icd_code,
        }

        es_record = {
            "_index": index_name,
            # "_type": "_doc",
            "_id": entity_id,
            "_source": content,
        }
        bulk_samples.append(es_record)

        if len(bulk_samples) == bulk_size:
            message, _ = bulk(
                es_conn, bulk_samples,
                index=index_name,
                raise_on_error=True,
            )
            bulk_samples = []

    if len(bulk_samples) > 0:
        message, _ = bulk(
            es_conn, bulk_samples,
            index=index_name,
            raise_on_error=True,
        )
        bulk_samples = []


if __name__ == "__main__":
    icd_data = pd.read_excel(
        "datasets/CBLUE_datasets/训练数据/CHIP-CDN/CHIP-CDN/国际疾病分类 ICD-10北京临床版v601.xlsx"
    )
    print(icd_data.head())

    icd_data.columns = ["icd_code", "entity_name"]
    print(icd_data.head())

    basic_tokenizer = BasicTokenizer(tokenize_chinese_chars=False,)

    es_config = {
        "ip": "localhost",
        "port": 9200,
    }
    # es_conn = Elasticsearch(
    #     [es_config["ip"]],
    #     http_auth=None,
    #     port=es_config["port"],
    # )
    es_conn = Elasticsearch("http://localhost:9200")
    print(es_conn.indices)

    index_name = "chip-cdn-0120"
    up_index_for_icd_el(icd_data, es_conn, index_name, bulk_size=1000)
