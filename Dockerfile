# RAGForge Goa backend. Indices are committed to the repo via Git LFS
# (data/processed/), so this image never runs scripts/build_index.py --
# startup only loads precomputed artifacts, matching the "no recomputation
# at request time" requirement.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY src/ src/
COPY configs/ configs/
COPY data/processed/ data/processed/
COPY data/raw/ data/raw/

ENV RAGFORGE_CONFIG=configs/production.yaml
EXPOSE 8420

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8420"]
