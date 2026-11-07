# One image for the API, the dashboard and the seeder; compose picks the command.
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime, which the slim image omits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install from requirements.txt minus what only notebooks, experiments and dev
# tooling use. The served path never imports torch, mlflow, jupyter and the like,
# and skipping them keeps the image to a few hundred MB instead of several GB.
# uv downloads and installs the wheels in parallel, which is far faster than pip on a slow
# connection. The long timeout is for the large wheels (scipy, lightgbm).
RUN pip install --no-cache-dir uv==0.13.7
ENV UV_HTTP_TIMEOUT=120

COPY requirements.txt .
RUN grep -vE '^(torch|mlflow|jupyter|ipykernel|xgboost|boto3|lime|fairlearn|imbalanced-learn|polars|black|isort|ruff|pytest|pytest-cov)==' requirements.txt > runtime.txt \
    && uv pip install --system --no-cache -r runtime.txt

COPY backend ./backend
COPY frontend ./frontend
COPY .streamlit ./.streamlit

ENV PYTHONUNBUFFERED=1
EXPOSE 8014 8501
