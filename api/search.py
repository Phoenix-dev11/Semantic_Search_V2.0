import numpy as np
from fastapi import APIRouter, HTTPException, Query, FastAPI
from database.database import VectorDB, async_session, SearchQuery, Companies, Products
from database.database import SearchResult as ORMSearchResult
from database.schemas import SearchRequest, SearchResult as SearchResultSchema, Product, ProductSpecifications
from sqlalchemy import select, or_, and_, func
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
import uuid
from datetime import datetime, date
import pandas as pd
import asyncio
from difflib import SequenceMatcher
import json

load_dotenv()

# Create FastAPI app for Vercel
# app = FastAPI()

# Create router for the search functionality
router = APIRouter()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def is_list_query(query_text: str) -> bool:
    """
    Check if the query is asking for a list of companies/products
    """
    list_keywords = [
        "list", "show", "all", "every", "companies", "products",
        "manufacturers", "suppliers", "vendors", "providers", "firms",
        "businesses", "列出", "显示", "所有", "全部", "公司", "产品", "制造商", "供应商", "厂商"
    ]

    query_lower = query_text.lower()
    return any(keyword in query_lower for keyword in list_keywords)


def is_all_industries_query(query_text: str) -> bool:
    """
    Check if the query wants to search across all industries (no industry filter)
    """
    all_industries_keywords = [
        "without industry", "all industries", "any industry",
        "no industry filter", "across industries", "不限行业", "所有行业", "任何行业",
        "跨行业", "全行业"
    ]

    query_lower = query_text.lower()
    return any(keyword in query_lower for keyword in all_industries_keywords)


def is_ranking_query(query_text: str) -> tuple[bool, str, str]:
    """
    Check if the query is asking for ranking by revenue or capital
    Returns: (is_ranking, field, order)
    """
    query_lower = query_text.lower()

    # Check for revenue ranking
    if any(keyword in query_lower
           for keyword in ["revenue", "income", "sales", "收入", "营收"]):
        if any(keyword in query_lower for keyword in
               ["largest", "highest", "biggest", "top", "最大", "最高", "最大"]):
            return True, "revenue", "desc"
        elif any(keyword in query_lower
                 for keyword in ["smallest", "lowest", "least", "最小", "最低"]):
            return True, "revenue", "asc"

    # Check for capital ranking
    if any(keyword in query_lower
           for keyword in ["capital", "capital_amount", "资金", "资本"]):
        if any(keyword in query_lower for keyword in
               ["largest", "highest", "biggest", "top", "最大", "最高", "最大"]):
            return True, "capital_amount", "desc"
        elif any(keyword in query_lower
                 for keyword in ["smallest", "lowest", "least", "最小", "最低"]):
            return True, "capital_amount", "asc"

    # Note: delivery time ranking removed as delivery_time field doesn't exist in current schema

    return False, "", ""


async def analyze_query(query_text: str) -> Dict[str, Any]:
    """
    Analyze natural language query using OpenAI to extract:
    - country: str | null (if mentioned in query)
    - weight: {certification: float, technology: float, standard: float}
    - query_text: short version for embedding
    - is_list_query: boolean indicating if this is a list query
    """
    print(f"      🤖 Calling OpenAI with query: '{query_text}'")

    # Check if it's a list query first
    is_list = is_list_query(query_text)
    is_all_industries = is_all_industries_query(query_text)
    is_ranking, ranking_field, ranking_order = is_ranking_query(query_text)
    print(f"      📋 Is list query: {is_list}")
    print(f"      🌐 Is all industries query: {is_all_industries}")
    print(
        f"      📊 Is ranking query: {is_ranking} (field: {ranking_field}, order: {ranking_order})"
    )

    # if is_list:
    #     # For list queries, extract country and return simple weights
    #     country_keywords = {
    #         "us": "US",
    #         "usa": "US",
    #         "america": "US",
    #         "united states": "US",
    #         "china": "CN",
    #         "chinese": "CN",
    #         "taiwan": "TW",
    #         "taipei": "TW",
    #         "japan": "JP",
    #         "japanese": "JP",
    #         "korea": "KR",
    #         "south korea": "KR",
    #         "korean": "KR",
    #         "germany": "DE",
    #         "german": "DE",
    #         "france": "FR",
    #         "french": "FR",
    #         "uk": "GB",
    #         "britain": "GB",
    #         "british": "GB",
    #         "united kingdom": "GB",
    #         "canada": "CA",
    #         "canadian": "CA",
    #         "australia": "AU",
    #         "australian": "AU",
    #         "india": "IN",
    #         "indian": "IN",
    #         "美国": "US",
    #         "中国": "CN",
    #         "台湾": "TW",
    #         "日本": "JP",
    #         "韩国": "KR",
    #         "德国": "DE",
    #         "法国": "FR",
    #         "英国": "GB",
    #         "加拿大": "CA",
    #         "澳大利亚": "AU",
    #         "印度": "IN"
    #     }

    #     query_lower = query_text.lower()
    #     country = None
    #     for keyword, country in country_keywords.items():
    #         if keyword in query_lower:
    #             country = country
    #             break

    #     return {
    #         "country": country,
    #         "weight": {
    #             "standard": 0.33,
    #             "certification": 0.34,
    #             "technology": 0.33,
    #         },
    #         "query_text": query_text,
    #         "is_list_query": True,
    #         "is_all_industries": is_all_industries,
    #         "is_ranking": is_ranking,
    #         "ranking_field": ranking_field,
    #         "ranking_order": ranking_order
    #     }

    # For semantic queries, use OpenAI analysis
    prompt = f"""
    Analyze the following search query and extract structured semantic information.

    Query: "{query_text}"

    Follow these exact rules:

    1. **Country Extraction**
    • Identify a country mentioned in the query and convert it to its ISO 2-letter country code.
    • If no country is mentioned, set "country" to null.
    • Use the following mappings:
        - United States, US, USA, America -> "US"
        - China, Chinese -> "CN"
        - Taiwan, Taipei -> "TW"
        - Japan, Japanese -> "JP"
        - South Korea, Korea, Korean -> "KR"
        - Germany, German -> "DE"
        - France, French -> "FR"
        - United Kingdom, UK, Britain, British -> "GB"
        - Canada, Canadian -> "CA"
        - Australia, Australian -> "AU"
        - India, Indian -> "IN"

    2. **Weights**
    • Assign weights for three aspects: "certification", "technology", and "standard".
    • These represent the relative importance or focus of each concept in the query.
    • If the query includes phrases like "without certification", "no certification", or "uncertified", set the certification weight to 0.
    • If none of the three concepts are mentioned or implied, use balanced defaults:
        certification = 0.33  
        technology = 0.33  
        standard = 0.34
    • All weights must be non-negative floats and must sum exactly to 1.0.
    3. **Short Query Text**
    • Create a concise, meaningful query_text (5–15 words) summarizing the intent of the query.
    • This text will be used for embedding generation.

    4. **List Query**
    • Set "is_list_query" to true if the query explicitly refers to plural or list-type searches 
        (e.g., "list of companies", "manufacturers in Germany", "suppliers", "distributors").
    • Otherwise, set it to false.

    5. **Industry Scope**
    • Set "is_all_industries" to true **only if the query explicitly mentions or implies coverage across all industries** 
        (e.g., "all industries", "cross-industry", "no specific industry", "industry-agnostic","without industry", "all industry", "any industry", "no industry filter", "across industries").
    • Queries like "all companies", "all suppliers", or "all manufacturers" do **NOT** imply all industries — they should be false.
    • Default is false.

    Respond ONLY with valid JSON in this exact format (no explanations or extra text):

    {{
        "country": "ISO_CODE or null",
        "weight": {{
            "certification": 0.0,
            "technology": 0.0,
            "standard": 0.0
        }},
        "query_text": "short meaningful text",
        "is_list_query": true,
        "is_all_industries": false
    }}
    """

    def _sync_analyze():
        response = openai_client.chat.completions.create(model="gpt-4o",
                                                         messages=[{
                                                             "role":
                                                             "user",
                                                             "content":
                                                             prompt
                                                         }],
                                                         temperature=0.1)
        return response.choices[0].message.content.replace("```json",
                                                           "").replace(
                                                               "```", "")

    try:
        result = await asyncio.to_thread(_sync_analyze)
        print(f"      ✅ OpenAI response: {result}")
        parsed = json.loads(result)

        # Validate and normalize weights
        weights = parsed.get("weight", {})
        cert_w = max(0, min(1, weights.get("certification", 0.33)))
        tech_w = max(0, min(1, weights.get("technology", 0.33)))
        standard_w = max(0, min(1, weights.get("standard", 0.34)))

        # Normalize to sum to 1.0
        total = cert_w + tech_w + standard_w
        if total > 0:
            cert_w /= total
            tech_w /= total
            standard_w /= total
        else:
            cert_w = tech_w = standard_w = 1 / 3

        parsed["weight"] = {
            "certification": round(cert_w, 3),
            "technology": round(tech_w, 3),
            "standard": round(standard_w, 3)
        }

        # Ensure is_all_industries is included
        parsed["is_all_industries"] = parsed.get("is_all_industries",
                                                 is_all_industries)
        parsed["is_ranking"] = is_ranking
        parsed["ranking_field"] = ranking_field
        parsed["ranking_order"] = ranking_order

        return parsed
    except Exception as e:
        print(f"      ❌ OpenAI analysis failed: {e}")
        # Fallback analysis
        is_ranking, ranking_field, ranking_order = is_ranking_query(query_text)
        return {
            "country": None,
            "weight": {
                "certification": 0.33,
                "technology": 0.33,
                "standard": 0.34
            },
            "query_text": query_text[:50],  # Truncate if too long
            "is_list_query": is_list_query(query_text),
            "is_all_industries": is_all_industries_query(query_text),
            "is_ranking": is_ranking,
            "ranking_field": ranking_field,
            "ranking_order": ranking_order
        }


async def get_available_industries() -> List[str]:
    """Get all unique industry categories from the database"""
    async with async_session() as session:
        result = await session.execute(
            select(Companies.industry_category).distinct())
        industries = []
        for row in result.fetchall():
            if row[0]:  # If industry_category is not None
                if isinstance(row[0], list):
                    industries.extend(row[0])  # Flatten arrays
                else:
                    industries.append(row[0])
        return list(set(industries))  # Remove duplicates


async def list_companies_and_products(
        industry_category: Optional[str],
        country: Optional[str],
        top_k: Optional[int] = 50,
        ranking_field: Optional[str] = None,
        ranking_order: str = "desc") -> List[Dict[str, Any]]:
    """
    Get a list of companies and products without semantic search
    Used for queries like "list all companies in US"
    """
    print(f"      📋 Listing companies and products...")
    print(f"         Industry: {industry_category}")
    print(f"         Country: {country}")
    print(
        f"         Limit: {top_k if top_k is not None else 'No limit (all results)'}"
    )

    async with async_session() as session:
        # Build query to get companies with their products
        query = select(
            Companies.id, Companies.name, Companies.industry_category,
            Companies.country, Companies.capital_amount, Companies.revenue,
            Companies.company_certification_documents, Companies.patent_count,
            Products.id.label("product_id"), Products.product_name,
            Products.main_raw_materials, Products.product_standard,
            Products.technical_advantages).select_from(
                Companies.__table__.join(Products.__table__,
                                         Companies.id == Products.company_id))

        # Add filters
        filters = []
        if industry_category:
            # Use PostgreSQL array contains operator for industry_category (ARRAY field)
            filters.append(
                Companies.industry_category.op('@>')([industry_category]))
        if country:
            filters.append(Companies.country == country)

        if filters:
            query = query.where(and_(*filters))

        # Add ranking if specified
        if ranking_field:
            if ranking_field == "revenue":
                if ranking_order == "desc":
                    query = query.order_by(Companies.revenue.desc())
                else:
                    query = query.order_by(Companies.revenue.asc())
            elif ranking_field == "capital_amount":
                if ranking_order == "desc":
                    query = query.order_by(Companies.capital_amount.desc())
                else:
                    query = query.order_by(Companies.capital_amount.asc())
            # Note: delivery_time field removed as it doesn't exist in current schema

        # Only apply limit if top_k is specified
        if top_k is not None:
            query = query.limit(top_k)

        # Execute query
        print(f"      📊 Executing list query...")
        print(f"      🔍 SQL Query: {query}")
        result = await session.execute(query)
        rows = result.fetchall()
        print(f"      ✅ Found {len(rows)} companies with products")

        # Debug: Show what was found
        if len(rows) == 0 and industry_category:
            print(
                f"      🔍 DEBUG: No results found for industry '{industry_category}'"
            )
            # Check if any companies have this industry
            debug_result = await session.execute(
                select(Companies.name, Companies.industry_category))
            debug_companies = debug_result.fetchall()
            print(f"      🔍 DEBUG: All companies in database:")
            for company in debug_companies:
                print(
                    f"         - {company.name}: {company.industry_category}")

        # Group by company
        companies_dict = {}
        for row in rows:
            company_id = row.id
            if company_id not in companies_dict:
                companies_dict[company_id] = {
                    "company_name": row.name,
                    "industry_category": row.industry_category,
                    "country": row.country,
                    "capital_amount": row.capital_amount,
                    "revenue": row.revenue,
                    "company_certification_documents":
                    row.company_certification_documents,
                    "patent_count": row.patent_count,
                    "products": []
                }

            # Add product to company with detailed structure
            if row.product_id:
                # Create product specifications (using available data)
                product_specs = {
                    "Dimensions": "Standard",  # Default since not in DB
                    "Prediction": 0.85,  # Default prediction score
                    "Materials": row.main_raw_materials
                }

                companies_dict[company_id]["products"].append({
                    "product_name":
                    row.product_name,
                    "main_raw_materials":
                    row.main_raw_materials,
                    "product_specifications":
                    product_specs,
                    "technical_advantages":
                    row.technical_advantages,
                    "product_standard":
                    row.product_standard
                })

        # Convert to list and sort by company name
        companies_list = list(companies_dict.values())
        # Preserve DB order when ranking is applied; otherwise sort by name for stable UX
        if not ranking_field:
            companies_list.sort(key=lambda x: x["company_name"])

        print(f"      🎯 Returning {len(companies_list)} companies")
        return companies_list


async def create_query_embedding(text: str) -> List[float]:
    """Create embedding for query text"""

    def _sync_embed():
        response = openai_client.embeddings.create(
            model="text-embedding-3-small", input=text)
        return response.data[0].embedding

    return await asyncio.to_thread(_sync_embed)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a_np = np.array(a)
    b_np = np.array(b)

    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


async def vector_search(query_embedding: List[float],
                        industry_category: Optional[str],
                        country: Optional[str],
                        top_k: int = 10) -> List[Dict[str, Any]]:
    """Perform vector search with industry and country filtering"""
    print(
        f"      🔍 Searching with industry='{industry_category}', country='{country}', top_k={top_k}"
    )
    async with async_session() as session:
        # Build base query with joins
        query = select(VectorDB.id, VectorDB.embedding, VectorDB.metadata_json,
                       Companies.name.label("company_name"),
                       Companies.industry_category, Companies.country,
                       Companies.capital_amount, Companies.revenue,
                       Companies.company_certification_documents,
                       Companies.patent_count, Products.product_name,
                       Products.technical_advantages,
                       Products.product_standard,
                       Products.main_raw_materials).select_from(
                           VectorDB.__table__.join(
                               Companies.__table__,
                               VectorDB.company_id == Companies.id).join(
                                   Products.__table__,
                                   VectorDB.product_id == Products.id))

        # Add filters
        filters = []
        if industry_category:
            # Use PostgreSQL array contains operator for industry_category (ARRAY field)
            filters.append(
                Companies.industry_category.op('@>')([industry_category]))
        if country:
            filters.append(Companies.country == country)

        if filters:
            query = query.where(and_(*filters))

        # Execute query
        print(f"      📊 Executing database query...")
        result = await session.execute(query)
        rows = result.fetchall()
        print(f"      ✅ Found {len(rows)} raw database rows")

        # Calculate similarities and rank
        results = []
        for i, row in enumerate(rows):
            similarity = cosine_similarity(query_embedding, row.embedding)
            print(
                f"         Row {i+1}: {row.company_name} - similarity: {similarity:.3f}"
            )
            results.append({
                "vector_id": row.id,
                "company_name": row.company_name,
                "industry_category": row.industry_category,
                "country": row.country,
                "capital_amount": row.capital_amount,
                "revenue": row.revenue,
                "company_certification_documents":
                row.company_certification_documents,
                "patent_count": row.patent_count,
                "product_name": row.product_name,
                "technical_advantages": row.technical_advantages,
                "product_standard": row.product_standard,
                "main_raw_materials": row.main_raw_materials,
                "metadata": row.metadata_json,
                "similarity": similarity
            })

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        print(f"      🎯 Returning top {min(top_k, len(results))} results")
        return results[:top_k]


def calculate_weighted_score(result: Dict[str, Any],
                             weights: Dict[str, float]) -> float:
    """Calculate weighted score based on metadata and weights"""
    metadata = result.get("metadata", {})

    # Base similarity score
    base_score = result.get("similarity", 0)

    # Certification score (based on number of certifications)
    cert_count = len(metadata.get("certifications", []))
    cert_score = min(cert_count / 5.0, 1.0)  # Normalize to 0-1

    # Technology score (based on technical_advantages length)
    tech_adv = metadata.get("technical_advantages", "")
    tech_score = min(len(tech_adv) / 200.0, 1.0)  # Normalize to 0-1

    # Delivery score (default to 0.5 since delivery_time field not available)
    standard_score = 0.5  # Neutral score since field not available standard

    # Data completeness score
    data_score = metadata.get("data_score", 0.5)

    # Calculate weighted score
    weighted_score = (
        base_score * 0.4 +  # 40% similarity
        cert_score * weights["certification"] * 0.2 +
        tech_score * weights["technology"] * 0.2 + standard_score *
        weights["standard"] * 0.2) * data_score  # Multiply by data quality

    return min(weighted_score, 1.0)


@router.get("/debug/industries")
async def debug_industries():
    """Debug endpoint to check what industries are stored in the database"""
    async with async_session() as session:
        # Get all companies with their industry categories
        result = await session.execute(
            select(Companies.name, Companies.industry_category))
        companies = result.fetchall()

        # Get all unique industries
        all_industries = await get_available_industries()

        return {
            "total_companies":
            len(companies),
            "all_industries":
            sorted(all_industries),
            "companies_with_industries": [{
                "name":
                company.name,
                "industries":
                company.industry_category
            } for company in companies]
        }


@router.post("/search")
async def search(request: SearchRequest):
    """
    Semantic search endpoint
    """
    print(f"🔍 SEARCH START: Query='{request.query_text}'")
    print(f"   Industry: {request.industry_category}")
    print(f"   Top K: {request.top_k}")

    try:
        # Step 1: Analyze query for country and weights only
        print("\n🧠 Step 1: Analyzing query...")
        print(f"   Query text: {request.query_text}")
        analysis = await analyze_query(request.query_text)
        print(f"   Analysis: {analysis}")
        country = analysis["country"]
        weights = analysis["weight"]
        query_text = analysis["query_text"]
        is_list_query = analysis.get("is_list_query", False)
        is_all_industries = analysis.get("is_all_industries", False)
        is_ranking = analysis.get("is_ranking", False)
        ranking_field = analysis.get("ranking_field", "")
        ranking_order = analysis.get("ranking_order", "desc")
        print(f"   ✅ Analysis result:")
        print(f"      Location: {country}")
        print(f"      Weights: {weights}")
        print(f"      Query text: {query_text}")
        print(f"      Is list query: {is_list_query}")
        print(f"      Is all industries: {is_all_industries}")
        print(
            f"      Is ranking: {is_ranking} (field: {ranking_field}, order: {ranking_order})"
        )

        # Step 2: Handle industry matching
        if is_all_industries:
            print("\n🌐 Step 2: All industries query - no industry filter")
            matched_industry_category = None
        else:
            available_industries = await get_available_industries()
            print(f"   Available industries: {available_industries}")
            matched_industry_category = request.industry_category
            print(f"   ✅ Using industry: {matched_industry_category}")

            # Debug: Check if the requested industry exists
            if matched_industry_category and matched_industry_category not in available_industries:
                print(
                    f"   ⚠️ WARNING: Requested industry '{matched_industry_category}' not found in available industries!"
                )
                print(
                    f"   🔍 Similar industries: {[ind for ind in available_industries if matched_industry_category.lower() in ind.lower() or ind.lower() in matched_industry_category.lower()]}"
                )

        # Check if this is a list query
        if is_list_query or is_ranking:
            print("\n📋 Step 3: Processing list query (no semantic search)...")
            # For list queries, don't apply top_k limit - show all results
            top_k_for_list = None if is_list_query else request.top_k
            companies_list = await list_companies_and_products(
                matched_industry_category, country, top_k_for_list,
                ranking_field if is_ranking else None,
                ranking_order if is_ranking else "desc")

            # Convert to search results format
            search_results = []
            vector_results = []  # Empty for list queries
            for i, company in enumerate(companies_list):
                for j, product in enumerate(company["products"]):
                    # Create product object (matching updated schema)
                    product_obj = Product(
                        product_name=product["product_name"],
                        main_raw_materials=product["main_raw_materials"],
                        product_standard=product["product_standard"],
                        technical_advantages=product["technical_advantages"])

                    # Create search result with new format
                    search_result = SearchResultSchema(
                        company_name=company["company_name"],
                        industry_category=company["industry_category"],
                        country=company["country"],
                        capital_amount=company["capital_amount"],
                        revenue=company["revenue"],
                        company_certification_documents=company[
                            "company_certification_documents"],
                        product=product_obj,
                        patent_count=company[
                            "patent_count"],  # Use integer directly
                        completeness_score=100,  # List queries show all data
                        semantic_score=
                        1.0,  # No semantic scoring for list queries
                        total_score=100)
                    search_results.append(search_result)

            print(
                f"   ✅ Found {len(search_results)} products from {len(companies_list)} companies"
            )
        else:
            # Step 3: Create query embedding
            print("\n🤖 Step 3: Creating query embedding...")
            query_embedding = await create_query_embedding(query_text)
            print(f"   ✅ Embedding created, dimension: {len(query_embedding)}")

            # Step 4: Vector search
            print("\n🔍 Step 4: Performing vector search...")
            vector_results = await vector_search(query_embedding,
                                                 matched_industry_category,
                                                 country, request.top_k)
            print(f"   ✅ Found {len(vector_results)} vector results")

            # Step 5: Calculate weighted scores and rank
            print("\n📊 Step 5: Calculating weighted scores...")
            search_results = []
            for i, result in enumerate(vector_results):
                print(
                    f"   Processing result {i+1}: {result['company_name']} - {result['product_name']}"
                )
                weighted_score = calculate_weighted_score(result, weights)
                print(
                    f"      Similarity: {result['similarity']:.3f}, Weighted score: {weighted_score:.3f}"
                )

                # Create product object (matching updated schema)
                product_obj = Product(
                    product_name=result["product_name"],
                    main_raw_materials=result.get("main_raw_materials", []),
                    product_standard=result.get("product_standard", []),
                    technical_advantages=result.get("technical_advantages",
                                                    []))

                # Create search result with new format
                search_result = SearchResultSchema(
                    company_name=result["company_name"],
                    industry_category=result["industry_category"],
                    country=result["country"],
                    capital_amount=result.get("capital_amount", 0),
                    revenue=result.get("revenue", 0),
                    company_certification_documents=result.get(
                        "company_certification_documents", []),
                    product=product_obj,
                    patent_count=result.get("patent_count",
                                            0),  # Use integer directly
                    completeness_score=int(
                        result["metadata"].get("data_score", 0.5) * 100),
                    semantic_score=round(weighted_score, 3),
                    total_score=int(weighted_score * 100))
                search_results.append(search_result)

            # Sort by total_score descending
            search_results.sort(key=lambda x: x.total_score, reverse=True)
            print(f"   ✅ Ranked {len(search_results)} results")

        # Step 6: Save search query and results to database
        print("\n💾 Step 6: Saving search query and results to database...")
        query_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        print(f"   Query ID: {query_id}")

        async with async_session() as session:
            # Save search query
            print("   📝 Saving search query...")
            search_query = SearchQuery(id=query_id,
                                       query_text=request.query_text,
                                       filters=json.dumps({
                                           "industry_category":
                                           matched_industry_category,
                                           "country": country
                                       }),
                                       top_k=request.top_k,
                                       created_at=now)
            session.add(search_query)

            # Save search results
            print(f"   📊 Saving {len(search_results)} search results...")
            for rank, result in enumerate(search_results, 1):
                # Find corresponding vector result
                vector_result = next(
                    (vr for vr in vector_results
                     if vr["company_name"] == result.company_name), None)
                vector_id = vector_result[
                    "vector_id"] if vector_result else None

                search_result = ORMSearchResult(
                    id=str(uuid.uuid4()),
                    query_id=query_id,
                    company=result.company_name,
                    product=result.product.product_name,
                    completeness_score=result.completeness_score,
                    semantic_score=result.semantic_score,
                    total_score=result.total_score,
                    rank=rank,
                    vector_id=vector_id,
                    created_at=now)
                session.add(search_result)

            print("   💾 Committing search data...")
            await session.commit()
            print("   ✅ Search data saved successfully")

        print(f"\n🎉 SEARCH COMPLETE: Found {len(search_results)} results")
        return {
            # "query_id": query_id,
            # "analysis": analysis,
            # "matched_industry_category": matched_industry_category,
            "results": search_results,
            "total_results": len(search_results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Include the router in the app
# app.include_router(router)

# Export the app for Vercel
# handler = app
