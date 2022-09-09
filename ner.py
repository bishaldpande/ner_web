import copy
import os
from typing import List, Union

import spacy
from loguru import logger
from nepali_stemmer.stemmer import NepStemmer
from spacy.tokens import Doc
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# for auto coloring
label_map = {
    "Location": "LOC",
    "Organization": "ORG",
    "Person": "PER",
    "Date": "DATE",
    "Event": "EVENT",
}
path = os.environ.get(
    "MODEL_PATH",
    "/home/bishal/Downloads/models/xlm-lg1/xlm-roberta-large-everestner/checkpoint-5193/",
)
model = AutoModelForTokenClassification.from_pretrained(path)
# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
tokenizer = AutoTokenizer.from_pretrained(path)
pipe = pipeline("ner", model, tokenizer=tokenizer, aggregation_strategy="max")


def fix_token_len(texts, entiites):
    entiites = copy.deepcopy(entiites)
    curr = 0
    start, end = [], []
    for w in texts:
        start.append(curr)
        e = curr + len(w)
        end.append(e)
        curr = e + 1
    start = start[::-1]

    for ent in entiites:
        s = ent["start"]
        e = ent["end"]
        for i in start:
            if i - 1 <= s:
                if s - i > 1:
                    print(ent, start, end)
                    print(s, i)
                s = i
                break
        for j in end:
            if j >= e:
                e = j
                break
        ent["start"] = s
        ent["end"] = e
    return entiites


def visualize_prediction(texts: Union[List[str], str]):
    if isinstance(texts, list):
        text = " ".join(texts)
    elif isinstance(texts, str):
        text = texts.strip()
    nlp = spacy.blank("ne")
    doc2 = nlp(text)
    entities = pipe(text)
    entities = fix_token_len(text.split(), entities)
    ents = [
        doc2.char_span(
            ents["start"],
            ents["end"],
            label=label_map.get(ents["entity_group"], ents["entity_group"]),
        )
        for ents in entities
    ]
    if None in ents:
        print("some issue are here")
        print(text)
        ents = [e for e in ents if e]
    if ents:
        try:
            doc2.ents = ents
        except:
            pass
    return doc2


nepstem = NepStemmer()


def predict(text: str, preprocess: bool = False):
    text = nepstem.stem(text)
    if preprocess:
        logger.debug("Normalizing text")
        trans = str.maketrans(
            {"श": "स", "ष": "स", "ी": "ि", "ब": "व", "ू": "ु", "ऊ": "उ", "ई": "इ"}
        )
        text = text.translate(trans)

    if len(tokenizer.tokenize(text)) > 500:
        texts = text.split("।")
    else:
        texts = [text]
    texts = list(filter(bool, texts))
    docs = [visualize_prediction(text.strip()) for text in texts]
    return docs


def visualize(docs):
    return spacy.displacy.render(docs, style="ent")
