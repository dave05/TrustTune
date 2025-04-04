from pydantic import BaseModel, Field
from typing import List, Optional

class BatchUpdateRequest(BaseModel):
    scores: List[float] = Field(..., description="List of prediction scores")
    labels: List[int] = Field(..., description="List of true labels") 