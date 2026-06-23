import pydantic
from typing import Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as st_models


class MLXEmbedder:
    """
    Thin wrapper around SentenceTransformer for Gemma-style embedding models.

    Handles instruction-prefix injection (document / query) so call sites can
    use prompt_name="document" or prompt_name="query" identically to before.
    mlx_lm.load is NOT used here — embedding models have SentenceTransformer
    pooling layers (dense.*.weight) that mlx_lm rejects as unknown parameters.
    """

    _PREFIXES: dict = {
        "document":        "Represent this document for retrieval: ",
        "passage":         "Represent this document for retrieval: ",
        "Retrieval-query": "Represent this query for searching relevant passages: ",
        "query":           "Represent this query for searching relevant passages: ",
    }

    def __init__(self, model_path: str):
        transformer = st_models.Transformer(model_path)
        pooling     = st_models.Pooling(
            transformer.get_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        self._st = SentenceTransformer(modules=[transformer, pooling])

    def encode(
        self,
        texts,
        prompt_name:       str  = None,
        show_progress_bar: bool = False,
        batch_size:        int  = None,
    ) -> np.ndarray:
        prefix = self._PREFIXES.get(prompt_name, "")
        if isinstance(texts, str):
            texts = prefix + texts
        else:
            texts = [prefix + t for t in texts]

        kwargs = {"show_progress_bar": show_progress_bar}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        return self._st.encode(texts, **kwargs)

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