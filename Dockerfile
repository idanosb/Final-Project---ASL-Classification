FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt || true

COPY . .

CMD ["python", "baseLine_model.py"]