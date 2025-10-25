from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Union


#Companies
class CompanyRequest(BaseModel):
    name: str
    industry_category: List[str]
    country: str
    capital_amount: float
    revenue: float
    company_certification_documents: List[str]
    patent_count: int


#Products
class ProductRequest(BaseModel):
    company_id: str
    product_name: str
    main_raw_materials: List[str]
    product_standard: List[str]
    technical_advantages: List[str]


#VectorDB
class VectorDBRequest(BaseModel):
    product_id: str
    company_id: str
    embedding: List[float]
    metadata_json: dict


#Search
class SearchRequest(BaseModel):
    # Required primary inputs
    query_text: str
    industry_category: str
    top_k: int = 5


# class NumericGap(BaseModel):
#     lead_time: Optional[str] = None
#     quality: Optional[str] = None
#     capacity: Optional[str] = None


class ProductSpecifications(BaseModel):
    Dimensions: str
    Prediction: float
    Materials: str


class Product(BaseModel):
    product_name: str
    main_raw_materials: List[str]
    product_standard: List[str]
    technical_advantages: List[str]


class SearchResult(BaseModel):
    company_name: str
    industry_category: List[str]
    country: str
    capital_amount: float
    revenue: float
    company_certification_documents: List[str]
    product: Product
    patent_count: int
    completeness_score: int
    semantic_score: float
    total_score: int


#Feedback
class FeedbackRequest(BaseModel):
    query_id: str
    result_id: str
    action_type: str  # "keep", "reject", "compare"


class FeedbackResponse(BaseModel):
    status: str  # "success" or "failure"
    message: str
