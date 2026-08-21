# RAGForge Goa backend. Indices are committed to the repo via Git LFS
# (data/processed/), so this image never runs scripts/build_index.py --
# startup only loads precomputed artifacts, matching the "no recomputation
# at request time" requirement.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git-lfs \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch explicitly, BEFORE sentence-transformers pulls in a
# default (CUDA-bundled, ~5x larger, higher RSS) build from PyPI. This
# repo never uses a GPU; the default wheel would only cost image size and
# memory for nothing, which matters at Railway's 1GB free-trial RAM ceiling.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding + reranker model weights into the image at build time.
# Without this, the first request in a fresh container would silently
# download ~175MB from HuggingFace Hub over the network -- a runtime
# dependency and failure mode ("HF unreachable" / rate-limited) that has no
# business existing when the model names are fixed and known at build time.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/msmarco-MiniLM-L6-cos-v5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY api/ api/
COPY src/ src/
COPY configs/ configs/
COPY data/processed/ data/processed/
COPY data/raw/ data/raw/

ENV RAGFORGE_CONFIG=configs/production.yaml
EXPOSE 8420

# Railway assigns a dynamic port via $PORT and routes to whatever the
# container actually binds -- a hardcoded --port would silently fail health
# checks in production even though `docker run -p 8420:8420` locally would
# look fine. Shell form so $PORT expands; falls back to 8420 for local
# `docker run` without -e PORT set.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8420}
