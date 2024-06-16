FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app
COPY starter.py /app/

RUN pip install --no-cache-dir pandas pyarrow scikit-learn

CMD ["python", "starter.py", "2023", "5"]


