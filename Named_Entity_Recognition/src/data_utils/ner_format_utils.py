

def conll2subword_conll(list_tokens, list_tags, tokenizer=None):
    # 为了支持英文， subword token打上 I-XXX标签
    # 这样在数据预处理时候处理subword问题，较为方便一些

    list_tokens_new, list_tags_new = [], []
    for tok, lab in zip(list_tokens, list_tags):
        tok_pieces = tokenizer.tokenize(tok)

        if lab.startswith("B-"):
            lab_pieces = [lab] + ["I-" + lab[2:], ] * (len(tok_pieces) - 1)
        else:
            lab_pieces = [lab, ] * len(tok_pieces)

        assert len(tok_pieces) == len(lab_pieces)
        list_tokens_new.extend(tok_pieces)
        list_tags_new.extend(lab_pieces)

    return list_tokens_new, list_tags_new


def conll2doccano_json(list_tokens, list_tags):
    # sent: str
    # ner labels: [start, end, mention, label]
    list_ent_spans = []
    tmp_span = []
    for i, (token, tag) in enumerate(zip(list_tokens, list_tags)):
        if tag == "0":
            if len(tmp_span) > 0:
                list_ent_spans.append(tmp_span)
                tmp_span = []

        if tag.startswith("B-"):
            ner_tag = tag[2:]

            if tmp_span:
                list_ent_spans.append(tmp_span)

            tmp_span = [i, i + 1, ner_tag]

        if tag.startswith("I-"):
            ner_tag = tag[2:]

            if (i > 0 and list_tags[i - 1] == 0) or i == 0:
                print("不符合conll format")
                tmp_span = [i, i + 1, ner_tag]

            if len(tmp_span) == 0:
                tmp_span = [i, i + 1, ner_tag]
            else:
                if ner_tag != tmp_span[2]:
                    list_ent_spans.append(tmp_span)
                    tmp_span = [i, i + 1, ner_tag]
                else:
                    tmp_span[1] = i + 1

    if tmp_span:
        list_ent_spans.append(tmp_span)

    # text_ = "".join(list_tokens)

    list_ent_spans_new = []
    for span_ in list_ent_spans:
        span_new_ = [
            span_[0],
            span_[1],
            list_tokens[span_[0]: span_[1]],
            span_[2]
        ]
        list_ent_spans_new.append(span_new_)

    return list_ent_spans_new
