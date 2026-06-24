import pydantic
from typing import Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as st_models




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