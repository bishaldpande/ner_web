FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN python -c "\
from transformers import AutoModel, AutoTokenizer;\
model_name = 'bishaldpande/Ner-xlm-roberta-base';\
AutoModel.from_pretrained(model_name);\
AutoTokenizer.from_pretrained(model_name);"

COPY . /app/
EXPOSE 5000
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
