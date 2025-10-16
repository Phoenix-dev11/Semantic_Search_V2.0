from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Union


#Companies
class CompanyRequest(BaseModel):
    name: str
    industry_category: str
    location: str
    capital_amount: int
    revenue: int
    company_certification_documents: str
    patent: bool
    delivery_time: int


#Products
class ProductRequest(BaseModel):
    company_id: str
    product_name: str
    main_raw_materials: str
    product_standard: str
    technical_advantages: str
    product_certifications: List[str]


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
    Product_Name: str
    Main_Raw_Materials: str
    Product_Specifications: ProductSpecifications
    Technical_Advantages: str
    Product_Certification_Materials: str


class SearchResult(BaseModel):
    Company_Name: str
    Industry_category: str
    Location: str
    capital_amount: int
    Revenue: int
    Company_certification_documents: str
    Product: Product
    Patent: bool
    Delivery_time: str
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
