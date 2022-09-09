from collections import Counter

from flask import Flask, render_template, request

from ner import predict, visualize

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def hello_world():
    result = ""
    show_name = request.form.get("show_name", 0)
    text = request.form.get("text", "")
    if text and request.method == "POST":
        string = request.args.get("preprocess", "")
        preprocess = string.lower() in {"true", "1", "t", "y", "yes"}
        docs = predict(text, preprocess)

        result = visualize(docs)
    return render_template("index.html", text=text, result=result, show_name=show_name)


@app.route("/api/v1/extract/", methods=["POST"])
def api():
    text = request.json.get("text", "")
    doc = predict(text)[0]
    data = {"text": text}
    data["entities"] = [
        {
            "key": ent.label_,
            "value": ent.text,
            "span": [ent.start_char, ent.end_char],
        }
        for ent in doc.ents
    ]
    data["named_entity_count"] = Counter(
        [f"{ent.text} ({ent.label_})" for ent in doc.ents]
    )
    data["entity_count"] = Counter([ent.label_ for ent in doc.ents])

    return data


if __name__ == "__main__":
    app.run()
