import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings # SETTINGS CLASS WHICH HAVE THE API KEYS 

# Variables

BATCH_SIZE = 50

# https://mteb-leaderboard.hf.space/models/google/gemini-embedding-2
# _GEMINI_DIM = 3072 # gemini ai embedding model dimension
_GEMINI_DIM = 768 # different dimension caues problem 


# https://mteb-leaderboard.hf.space/models/sentence-transformers/all-mpnet-base-v2

_FALLBACK_DIM = 768 # WHEN WE GET A RATE LIMITER ON EMBEDDING CALL weaker model

# Matryoshka Representation Learning (MRL),

_active_model = None # holds the loaded model instance 
_model_type: str | None = None # "gemini" or "fallback"

def _probe_gemini():
    """
    try one embed call to verify gemini is working, return model or none
    """

    # health check for gemini embedding model
    try :
        model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2", 
            google_api_key=settings.GOOGLE_API_KEY,
            output_dimensionality=_GEMINI_DIM,)

        # pass dim per call too: some langchain versions ignore the constructor arg
        vec = model.embed_query("test", output_dimensionality=_GEMINI_DIM)
        if len(vec) != _GEMINI_DIM:
            raise ValueError(f"Gemini returned dim {len(vec)}, expected {_GEMINI_DIM}")
        
        
        logfire.info(f"✅ Gemini Embedding Model is ready model = gemini-embedding-2, dim = {_GEMINI_DIM}")
        return model
    
    except Exception as e:
        logfire.error(f"Failed to initialize Gemini Embedding Model:{e}, using fallback model instead")
        return None

def _load_fallback():
    """
    load fallback model, return model or none
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name_or_path = "sentence-transformers/all-mpnet-base-v2")
    logfire.info("✅ Fallback Embedding Model is ready model = sentence-transformers/all-mpnet-base-v2, dim = 768 ")
    return model 

def _init():
    global _active_model, _model_type
    if _active_model is not None: 
        # if model is already loaded, then return it
        return 

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"
    return 


def get_embedding_dim() -> int:
    """
    return the vector dimension of the active model
    """

    _init()  # make sure the model probe has run so _model_type is set
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM



# other modeules cannot call this _ function ,its private to this module only 
def _embed_batch(batch : list[str]) -> list[list[float]]: 
    """ 
    embed a batch of list of texts using the embedding model, return list of embeddings list (list of floats)
    each element of list is converted into a list of float (vector)
    """
    # _init() # lazy-load the model on first use (no-op afterwards)
    if _model_type == "gemini":
        # if active model is gemini
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch, output_dimensionality=_GEMINI_DIM)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ["rate limit", "quota exceeded", "429"])
                
                if is_rate_limit and attempt < 3:
                    wait = 2**attempt
                    logfire.warning(f"Gemini embedding rate limit hit, retrying in {wait}s" 
                                    f"(attempt {attempt+1}/4)")
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise e
        # dedented: only reached after all retries are used up
        raise RuntimeError("Gemini embedding failed after 4 retries")
    else:
        # if fallback model we use the encode method of sentance transformers model 
        return _active_model.encode(batch, show_progress_bar=False).tolist()



# other modules can call this function without _
def embed_query(query :str) -> list[float]:
    """
    embed a single query string, return embedding list of floats(vectors)
    """
    _init() 
    embed_query_vector :list[float] = []
    if _model_type == "gemini":
        embed_query_vector = _active_model.embed_query(query, output_dimensionality=_GEMINI_DIM)   
    else:
        embed_query_vector = _active_model.encode([query], show_progress_bar=False).tolist()[0]

    return embed_query_vector

def embed_texts(texts : list[str]) -> list[list[float]]:
    """
    embed a list of texts, return list of embeddings list (list of floats)
    """
    _init() # lazy-load the model on first use (no-op afterwards)
    all_embeddings : list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        with logfire.span(f"Embed batch via {_model_type}   ", start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))

            
    return all_embeddings