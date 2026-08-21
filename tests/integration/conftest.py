import pytest

from src.embeddings.embedder import Embedder
from src.retrieval.index_store import IndexStore
from src.telemetry.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config("configs/development.yaml")


@pytest.fixture(scope="session")
def ragforge_store(cfg):
    return IndexStore(cfg.path(cfg.data.processed_dir) / "ragforge")


@pytest.fixture(scope="session")
def embedder(cfg):
    return Embedder(cfg.embeddings.model_name, device=cfg.embeddings.device, batch_size=cfg.embeddings.batch_size)
