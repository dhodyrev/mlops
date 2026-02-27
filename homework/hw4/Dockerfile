FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install pipenv

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

COPY score.py .

RUN mkdir -p /app/output

ENTRYPOINT ["python", "score.py"]