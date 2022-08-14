from flask import Flask, render_template, request
from loguru import logger

from ner import predict

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def hello_world():
    result = ""
    show_name = request.form.get("show_name", 0)
    text = request.form.get("text", "")
    if text and request.method == "POST":
        string = request.args.get("preprocess", "")
        preprocess = string.lower() in {"true", "1", "t", "y", "yes"}
        result = predict(text, preprocess)
    return render_template("index.html", text=text, result=result, show_name=show_name)


if __name__ == "__main__":
    app.run()
