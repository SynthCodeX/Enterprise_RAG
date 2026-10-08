"""
Live checks for the Gemini embedding model used by app/services/retrieval/embeddings.py.

Run from the project root:
    python -m pytest tests/test_gemini_embedding.py -v -s

Skipped when GEMINI_API_KEY is not set. These make real (cheap) embedding calls.
"""
import math

import pytest
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings
from app.services.retrieval import embeddings as emb

MODEL = "gemini-embedding-2"
DIM = emb._GEMINI_DIM

# Realistic RAG-style corpus: two topics (Kubernetes, NUMA/memory) plus one off-topic doc.
DOCS = {
    "k8s_pod": "A Kubernetes Pod is the smallest deployable unit and wraps one or more containers sharing network and storage.",
    "k8s_hpa": "The Horizontal Pod Autoscaler scales the number of pod replicas based on observed CPU utilization or custom metrics.",
    "numa": "NUMA systems give each CPU socket local memory, so accessing remote memory is slower than local memory.",
    "cooking": "To make a classic margherita pizza, top the dough with tomato sauce, mozzarella and fresh basil, then bake hot.",
}

# (query, expected best-matching doc key)
QUERIES = [
    ("How do I automatically scale my deployment under load?", "k8s_hpa"),
    ("What is a pod in Kubernetes?", "k8s_pod"),
    ("Why is accessing memory on another CPU socket slower?", "numa"),
]


@pytest.fixture(scope="module")
def model():
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is not set")
    return GoogleGenerativeAIEmbeddings(
        model=MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        output_dimensionality=DIM,
    )


@pytest.fixture(scope="module")
def doc_vectors(model):
    keys = list(DOCS)
    vecs = model.embed_documents([DOCS[k] for k in keys], output_dimensionality=DIM)
    return dict(zip(keys, vecs))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_query_embedding_has_expected_dimension(model):
    vec = model.embed_query("What is Kubernetes?", output_dimensionality=DIM)
    assert len(vec) == DIM
    assert all(isinstance(x, float) for x in vec)


def test_batch_embedding_returns_one_vector_per_text(doc_vectors):
    assert set(doc_vectors) == set(DOCS)
    assert all(len(v) == DIM for v in doc_vectors.values())


@pytest.mark.parametrize("query,expected", QUERIES)
def test_query_retrieves_most_relevant_doc(model, doc_vectors, query, expected):
    qv = model.embed_query(query, output_dimensionality=DIM)
    scores = {k: cosine(qv, v) for k, v in doc_vectors.items()}
    best = max(scores, key=scores.get)
    print(f"\n{query!r} -> {best} {scores}")
    assert best == expected, f"expected {expected}, got {best}: {scores}"


def test_related_docs_closer_than_unrelated(doc_vectors):
    same_topic = cosine(doc_vectors["k8s_pod"], doc_vectors["k8s_hpa"])
    cross_topic = cosine(doc_vectors["k8s_pod"], doc_vectors["cooking"])
    assert same_topic > cross_topic


def test_module_selects_gemini_not_fallback(monkeypatch):
    """The app's own probe must succeed; otherwise it silently uses the weaker fallback model."""
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is not set")
    assert emb._probe_gemini() is not None, "app probe failed; embeddings fall back to all-mpnet-base-v2"
