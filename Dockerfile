FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install .

EXPOSE 7860

CMD ["python", "inference.py"]
