import pydantic
from typing import Optional
from datetime import datetime
import numpy as np
import mlx.core as mx
from mlx_lm import load as _mlx_load


class MLXEmbedder:
    """
    Drop-in replacement for SentenceTransformer that runs fully on MLX.
    Loads any mlx-lm compatible model and produces fixed-size embeddings via
    mean pooling over the transformer body's last hidden states.
    """

    # Gemma embedding instruction prefixes (empty string = no prefix)
    _PREFIXES: dict = {
        "document":        "Represent this document for retrieval: ",
        "passage":         "Represent this document for retrieval: ",
        "Retrieval-query": "Represent this query for searching relevant passages: ",
        "query":           "Represent this query for searching relevant passages: ",
    }

    def __init__(self, model_path: str):
        self._model, self.tokenizer = _mlx_load(model_path)

    def encode(
        self,
        texts,
        prompt_name: str = None,
        show_progress_bar: bool = False,
        batch_size: int = None,
    ) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        prefix = self._PREFIXES.get(prompt_name, "")
        embeddings = []
        for text in texts:
            tokens = self.tokenizer.encode(prefix + text)
            ids = mx.array([tokens])
            out = self._model.model(ids)
            hidden = out[0] if isinstance(out, tuple) else out
            emb = hidden.mean(axis=1)   # [1, hidden_size]
            mx.eval(emb)
            embeddings.append(np.array(emb[0].tolist(), dtype=np.float32))

        result = np.stack(embeddings)
        return result[0] if single else result

class doc_payload(pydantic.BaseModel):
    '''
    This class insures data consistency. Not all metadata fields exist for all chunks. Therefore, they're defaulted to None
    '''
    form_type: Optional[str] = None
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    fiscal_year_end: Optional[datetime] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    item: Optional[str] = None
    # parent-document retrieval linkage
    doc_id: Optional[str] = None       # UUID of the full-document chunk
    parent_id: Optional[str] = None    # UUID of the header-level chunk (child chunks only)
    # level-3 enrichment
    description: Optional[str] = None  # LLM-generated summary; embed this instead of text

    # Clean and normalize strings automatically
    @pydantic.field_validator('form_type', 'company_name', 'ticker', 'section', 'subsection', 'item', mode="before")
    @classmethod
    def lowercase_string(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v