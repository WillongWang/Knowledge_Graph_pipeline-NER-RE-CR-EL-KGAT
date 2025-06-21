import json
import random

from elasticsearch import Elasticsearch
from tqdm import tqdm
from transformers import BasicTokenizer

import pandas as pd


def retrieve_by_es(entity_mention, es_conn, index_name="chip-cdn-0120", topk=200):
    must_conditions = [
        {
            "match": {
                "entity_name": entity_mention,
            }
        }
    ]
    doc = dict()
    doc["query"] = {
        "bool": {
            "must": must_conditions,
            "must_not": [],
        }
    }
    doc["size"] = topk

    list_retrieved_results = []
    try:
        results = es_conn.search(index=index_name, body=doc)
        # print(results['hits']['hits'])

        for res in results['hits']['hits']:
            list_retrieved_results.append(res["_source"])

    except Exception as e:
        print(e)

    return list_retrieved_results


def prepare_nli_dataset(data_dir=None,
                        neg_multiplier=4,
                        to_dir=None,
                        ):

    list_samples = json.load(
        open(data_dir, "r", encoding="utf-8")
    )

    df_data = []

    for samp in tqdm(list_samples):
        mention = samp["text"]
        normalized_result = samp["normalized_result"]
        ents = normalized_result.split("##")
        ents = [basic_tokenizer._clean_text(w.strip()) for w in ents]
        ents = [w for w in ents if w]

        # 正样本：
        for ent in ents:
            df_data.append(
                {
                    "text_a": mention,
                    "text_b": ent,
                    "label": 1,
                }
            )

        # 生成负样本：
        # 正负样本比例： 1： 3
        retrieved_samples = retrieve_by_es(
            mention,
            es_conn,
            index_name="chip-cdn-0120",
            topk=1000
        )
        if len(retrieved_samples) == 0:
            print(mention, retrieved_samples)
            continue

        count = 0
        seen_ents = []
        seen_ents.extend(ents)

        for samp in retrieved_samples[: len(ents) * neg_multiplier + len(ents) + 1]:

            if samp["entity_name"] in seen_ents:
                continue
            else:
                df_data.append(
                    {
                        "text_a": mention,
                        "text_b": samp["entity_name"],
                        "label": 0,
                    }
                )
                count += 1

                seen_ents.append(samp["entity_name"])

    df_data = pd.DataFrame(df_data)
    df_data.to_csv(to_dir, index=False, header=False, sep="\t")




if __name__ == "__main__":
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
    entity_mention = "上呼吸道"
    list_retrieved_results = retrieve_by_es(
        entity_mention,
        es_conn,
        index_name=index_name,
        topk=256
    )
    print(list_retrieved_results)

    basic_tokenizer = BasicTokenizer(tokenize_chinese_chars=False, )

    # 构造训练集：
    prepare_nli_dataset(data_dir="datasets/CBLUE_datasets/CHIP-CDN/CHIP-CDN/CHIP-CDN_train.json",
                        neg_multiplier=9,
                        to_dir="datasets/CBLUE_datasets/CHIP-CDN/training_data_0120/train.txt",
                        )

    # 构造验证集：
    prepare_nli_dataset(data_dir="datasets/CBLUE_datasets/CHIP-CDN/CHIP-CDN/CHIP-CDN_dev.json",
                        neg_multiplier=32,
                        to_dir="datasets/CBLUE_datasets/CHIP-CDN/training_data_0120/dev.txt",
                        )
