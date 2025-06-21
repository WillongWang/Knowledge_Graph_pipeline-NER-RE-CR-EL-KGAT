from elasticsearch import Elasticsearch


def create_index_for_icd_el(es_conn, index_name, doc_type_name="icd_name_code"):

    if es_conn.indices.exists(index=index_name):
        res = es_conn.indices.delete(index=index_name)
        print(res)

    new_mapping = {
            "properties": {
                "entity_name": {
                    "type": "text"
                },
                "icd_code": {
                    "type": "text"
                },
                "entity_id": {
                    "type": "keyword"
                },
            }
        }

    settings = {
        "mappings": new_mapping,
    }

    create_message = es_conn.indices.create(
        index=index_name, body=settings,
    )
    print(create_message)


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
    create_index_for_icd_el(es_conn, index_name, doc_type_name="icd_name_code")

