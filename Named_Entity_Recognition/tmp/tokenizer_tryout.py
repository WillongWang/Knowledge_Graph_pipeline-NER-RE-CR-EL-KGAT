from src.transformers import BertTokenizer

if __name__ == "__main__":


    tokenizer = BertTokenizer.from_pretrained(
        # "resources/chinese_bert_wwm_ext/",
        "bert-base-uncased",
        # do_lower_case=True
    )

    # sent = "苹果的Appstore收费太贵"
    sent = "London 1996-08-30"
    print(" ".join(tokenizer.tokenize(sent)))


