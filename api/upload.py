import pandas as pd
import uuid
import json
from fastapi import APIRouter, UploadFile, File, HTTPException, FastAPI
from database.database import Companies, Products, VectorDB, async_session, get_session
from database.schemas import CompanyRequest, ProductRequest, VectorDBRequest
from sqlalchemy import select
from typing import List, Dict, Any
import io
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import date, datetime
import numpy as np
import math
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

load_dotenv()

# # Create FastAPI app for Vercel
# app = FastAPI()

# Create router for the upload functionality
router = APIRouter()


def convert_chinese_number_to_int(value: str,
                                  unit_billions: bool = False) -> float:
    """
    Convert Chinese number strings to float values.
    Handles formats like "20億" -> 2000000000.0, "5千萬" -> 50000000.0, etc.
    
    Args:
        value: The Chinese number string to convert
        unit_billions: If True, divide result by 10^9 to store in billions unit
    """
    if not value or pd.isna(value):
        return 0

    value_str = str(value).strip()
    if not value_str:
        return 0

    # Remove any non-numeric and non-Chinese characters except decimal point
    import re

    # Extract number and unit
    match = re.match(r'([\d.]+)\s*([億万千百十]*)', value_str)
    if not match:
        try:
            result = float(value_str)
            return result / (10**9) if unit_billions else result
        except:
            return 0.0

    number_part = float(match.group(1))
    unit_part = match.group(2)

    # Chinese number multipliers
    multipliers = {
        '億': 100000000,  # 100 million
        '万': 10000,  # 10 thousand  
        '千': 1000,  # thousand
        '百': 100,  # hundred
        '十': 10,  # ten
    }

    multiplier = 1
    for unit in unit_part:
        if unit in multipliers:
            multiplier *= multipliers[unit]

    result = int(number_part * multiplier)

    # Convert to billions unit if requested
    if unit_billions:
        return result / (10**9)  # Use division for float result

    return result


def calculate_score(row: pd.Series) -> float:
    """
    Calculate score for data.
    Start with 1.0, minus 0.05 for each empty column.
    Handles scalars and list-like values safely.
    """
    score = 1.0
    empty_penalty = 0.05

    def is_empty(value: Any) -> bool:
        # None or NaN
        if value is None:
            return True
        try:
            if pd.isna(value):
                # pd.isna(list) raises, caught below
                return True
        except Exception:
            pass

        # List-like: empty or all items empty/whitespace
        if isinstance(value, (list, tuple, set)):
            if len(value) == 0:
                return True
            for item in value:
                if item is None:
                    continue
                s = str(item).strip()
                if s != "":
                    return False
            return True

        # Numpy arrays
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return True
            # Consider empty if all stringified items are empty
            return all(str(v).strip() == "" for v in value.flatten())

        # Scalar string/number
        try:
            return str(value).strip() == ""
        except Exception:
            return False

    # Check each column for empty values
    for column in row.index:
        value = row[column]
        if is_empty(value):
            score -= empty_penalty
    return max(0.0, score)  # Ensure score doesn't go below 0


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload Excel or CSV file containing company or product data.
    Automatically updates VectorDB for product data.
    Product Data: File must contain an industry category column (e.g., '產業別', 'industry_category').
    """
    print(f"🚀 UPLOAD START: Processing file {file.filename}")
    print(f"📋 Step 1: File validation and preparation")
    file_id = str(uuid.uuid4())

    # Check file extension
    filename = (file.filename or "").lower()
    file_extension = filename.split('.')[-1] if '.' in filename else ''
    print(f"📁 File extension: {file_extension}")

    # Read bytes once
    file_bytes = await file.read()

    # Decide parser by extension or content-type
    is_json = file_extension in ["json"] or (file.content_type
                                             and "json" in file.content_type)
    is_tabular = file_extension in ["csv", "xlsx", "xls"]

    if not (is_json or is_tabular):
        raise HTTPException(
            status_code=400,
            detail="Only CSV, Excel, or JSON files are supported")

    try:
        if is_json:
            print("📝 Parsing JSON file...")
            records = _parse_json_payload(file_bytes)
        else:
            print("📊 Parsing tabular file...")
            records = _parse_tabular_payload(file_bytes, file_extension)
        print(f"✅ Parsed {len(records)} records")
    except Exception as e:
        print(f"❌ Parse error: {e}")
        raise HTTPException(status_code=400,
                            detail=f"Failed to parse file: {e}")

    if not records:
        print("⚠️ No records found in file")
        return {
            "status": "ok",
            "message": "No records found",
            "created": 0,
            "updated": 0,
            "industry_categories": [],
            "total_industry_categories": 0
        }

    print(f"📋 Step 3: Environment and API validation")
    # Upsert into DB
    created_counts = {"companies": 0, "products": 0, "vectors": 0}
    updated_counts = {"companies": 0, "products": 0, "vectors": 0}

    # Track all industry categories across all records
    industry_categories = set()

    # Ensure OpenAI API key present if embeddings required
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        print("❌ OPENAI_API_KEY is not set")
        raise HTTPException(status_code=500,
                            detail="OPENAI_API_KEY is not set")
    else:
        print(f"✅ OpenAI API key found: {openai_api_key[:10]}...")

    # OpenAI embeddings model
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    print(f"🤖 Using embedding model: {embedding_model}")

    # Test network connectivity
    print("🌐 Testing network connectivity...")
    try:
        import socket
        socket.create_connection(("api.openai.com", 443), timeout=10)
        print("✅ Network connectivity to OpenAI API confirmed")
    except Exception as network_error:
        print(f"❌ Network connectivity test failed: {network_error}")
        print("🔧 This may cause embedding creation to fail")

    print(f"📋 Step 4: Database connection and transaction start")
    print("💾 Starting database operations...")

    # Process records in batches to avoid memory issues
    batch_size = 5  # Reduced batch size to prevent connection timeouts
    total_batches = (len(records) + batch_size - 1) // batch_size
    print(
        f"📊 Processing {len(records)} records in {total_batches} batches of {batch_size}"
    )
    print(
        f"⏱️ Estimated time: {total_batches * 1.5} minutes (1.5 min per batch)"
    )
    print(f"🔧 Using smaller batches to prevent database connection timeouts")

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(records))
        batch_records = records[start_idx:end_idx]

        print(
            f"\n📋 Step 5: Processing batch {batch_num + 1}/{total_batches} ({len(batch_records)} records)"
        )
        print(
            f"📈 Progress: {((batch_num + 1) / total_batches * 100):.1f}% complete"
        )

        # Add retry logic for database connection issues
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                async with async_session() as session:
                    for i, rec in enumerate(batch_records):
                        global_record_num = start_idx + i + 1
                        print(
                            f"\n📝 Processing record {global_record_num}/{len(records)} (batch {batch_num + 1})"
                        )
                        print(f"   Company: {rec.get('company_name', 'N/A')}")
                        print(f"   Product: {rec.get('product_name', 'N/A')}")
                        print(
                            f"   Industry: {rec.get('industry_category', 'N/A')}"
                        )

                        # Collect industry categories from this record
                        record_industries = rec.get('industry_category', [])
                        if isinstance(record_industries, list):
                            for industry in record_industries:
                                if industry and str(industry).strip():
                                    industry_categories.add(
                                        str(industry).strip())
                        elif record_industries and str(
                                record_industries).strip():
                            industry_categories.add(
                                str(record_industries).strip())

                        # Log industry categories being collected
                        if record_industries:
                            print(f"   🏭 Industries: {record_industries}")

                        print(
                            f"   📋 Step 5.{global_record_num}.1: Upserting company..."
                        )
                        # Upsert company
                        company_res, company_created = await _upsert_company(
                            session, rec)
                        if company_created:
                            created_counts["companies"] += 1
                            print(f"   ✅ Created company: {company_res.id}")
                        else:
                            updated_counts["companies"] += 1
                            print(f"   🔄 Updated company: {company_res.id}")

                        print(
                            f"   📋 Step 5.{global_record_num}.2: Processing products..."
                        )
                        # Get products for this record (handle multiple products separated by Chinese comma)
                        products = _parse_products(rec.get("product_name", ""))
                        print(
                            f"   📦 Found {len(products)} products: {products}")

                        # Create one product record per product
                        product_records = []
                        for product_idx, product_name in enumerate(products):
                            if not product_name or str(
                                    product_name).strip() == "":
                                continue

                            product_name = str(product_name).strip()
                            print(
                                f"   📋 Step 5.{global_record_num}.2.{product_idx + 1}: Processing product '{product_name}'"
                            )

                            # Create a copy of the record with this specific product
                            product_rec = rec.copy()
                            product_rec["product_name"] = product_name

                            # Upsert product
                            product_res, product_created = await _upsert_product(
                                session, company_res.id, product_rec)
                            if product_created:
                                created_counts["products"] += 1
                                print(
                                    f"   ✅ Created product: {product_res.id}")
                            else:
                                updated_counts["products"] += 1
                                print(
                                    f"   🔄 Updated product: {product_res.id}")

                            product_records.append(product_res)

                        print(
                            f"   📋 Step 5.{global_record_num}.3: Processing industries for vector creation..."
                        )
                        # Get industries for this record
                        industries = rec.get("industry_category", [])
                        if not industries:
                            print(
                                f"   ⚠️ No industries found, skipping vector creation"
                            )
                            continue

                        print(
                            f"   🏭 Found {len(industries)} industries: {industries}"
                        )

                        # Create one vector per industry (with all products in metadata)
                        for industry_idx, industry in enumerate(industries):
                            if not industry or str(industry).strip() == "":
                                continue

                            industry = str(industry).strip()
                            print(
                                f"   📋 Step 5.{global_record_num}.4.{industry_idx + 1}: Processing industry '{industry}' with {len(product_records)} products"
                            )

                            # Build metadata for this industry with all products
                            metadata, score = _build_metadata_for_industry_with_products(
                                rec, industry, product_records)
                            print(
                                f"   📈 Data score for industry '{industry}': {score}"
                            )

                            print(
                                f"   📋 Step 5.{global_record_num}.5.{industry_idx + 1}: Building embedding text for industry '{industry}'..."
                            )
                            # Create embedding text for this industry
                            embedding_text = _build_embedding_text_for_industry_with_products(
                                metadata, industry, product_records)
                            print(
                                f"   📝 Embedding text length: {len(embedding_text)} chars"
                            )

                            print(
                                f"   📋 Step 5.{global_record_num}.6.{industry_idx + 1}: Creating OpenAI embedding for industry '{industry}'..."
                            )
                            # Call OpenAI for embedding
                            try:
                                embedding = await _create_embedding_async(
                                    embedding_text, embedding_model,
                                    openai_api_key)
                                print(
                                    f"   ✅ Embedding created for industry '{industry}', dimension: {len(embedding)}"
                                )
                            except Exception as embedding_error:
                                print(
                                    f"   ❌ Embedding creation failed for industry '{industry}': {embedding_error}"
                                )
                                print(
                                    f"   🔧 Error type: {type(embedding_error).__name__}"
                                )
                                print(
                                    f"   🔧 Error details: {str(embedding_error)}"
                                )
                                raise

                            print(
                                f"   📋 Step 5.{global_record_num}.7.{industry_idx + 1}: Upserting vector for industry '{industry}'..."
                            )
                            # Upsert vector for this industry (no specific product_id)
                            vec_res, vec_created = await _upsert_vector_for_industry_only(
                                session,
                                company_id=company_res.id,
                                industry_category=industry,
                                embedding=embedding,
                                metadata=metadata)
                            if vec_created:
                                created_counts["vectors"] += 1
                                print(
                                    f"   ✅ Created vector for industry '{industry}': {vec_res.id}"
                                )
                            else:
                                updated_counts["vectors"] += 1
                                print(
                                    f"   🔄 Updated vector for industry '{industry}': {vec_res.id}"
                                )

                    print(
                        f"\n📋 Step 6: Committing batch {batch_num + 1} transaction..."
                    )
                    await session.commit()
                    print(
                        f"✅ Batch {batch_num + 1} transaction committed successfully"
                    )
                    break  # Success, exit retry loop

            except Exception as e:
                retry_count += 1
                print(
                    f"❌ Batch {batch_num + 1} database error (attempt {retry_count}/{max_retries}): {e}"
                )
                print(f"🔧 Error type: {type(e).__name__}")
                print(f"🔧 Error details: {str(e)}")

                if retry_count < max_retries:
                    print(f"🔄 Retrying batch {batch_num + 1} in 5 seconds...")
                    import asyncio
                    await asyncio.sleep(5)  # Wait 5 seconds before retry
                else:
                    print(
                        f"❌ Batch {batch_num + 1} failed after {max_retries} attempts"
                    )
                    try:
                        print("🔄 Attempting database rollback...")
                        await session.rollback()
                        print("✅ Database rollback completed")
                    except Exception as rollback_error:
                        print(f"❌ Rollback failed: {rollback_error}")
                        print(
                            f"🔧 Rollback error type: {type(rollback_error).__name__}"
                        )
                    raise

    print(f"📋 Step 7: Final summary and response preparation")
    # Update todos via return payload
    total_created = sum(created_counts.values())
    total_updated = sum(updated_counts.values())

    # Convert industry_categories set to sorted list for consistent output
    industry_categories_list = sorted(list(industry_categories))
    print(
        f"📊 Found {len(industry_categories_list)} unique industry categories: {industry_categories_list}"
    )

    print(f"🎉 UPLOAD COMPLETE:")
    print(f"   📁 File: {file.filename}")
    print(f"   📊 Records processed: {len(records)}")
    print(f"   🏢 Companies created: {created_counts['companies']}")
    print(f"   🏢 Companies updated: {updated_counts['companies']}")
    print(f"   📦 Products created: {created_counts['products']}")
    print(f"   📦 Products updated: {updated_counts['products']}")
    print(f"   🧮 Vectors created: {created_counts['vectors']}")
    print(f"   🧮 Vectors updated: {updated_counts['vectors']}")
    print(f"   🏭 Industry categories: {len(industry_categories_list)}")

    return {
        "status": "ok",
        "file_id": file_id,
        "created": created_counts,
        "updated": updated_counts,
        "total_created": total_created,
        "total_updated": total_updated,
        "industry_categories": industry_categories_list,
        "total_industry_categories": len(industry_categories_list),
    }


def _parse_tabular_payload(file_bytes: bytes,
                           file_extension: str) -> List[Dict[str, Any]]:
    buffer = io.BytesIO(file_bytes)
    if file_extension == "csv":
        df = pd.read_csv(buffer)
        # Normalize column names to lower snake for mapping
        df.columns = [str(c).strip() for c in df.columns]
        dataframes = [df]
        sheet_names = ["CSV"]
    else:
        # Read all sheets from Excel file
        excel_file = pd.ExcelFile(buffer)
        sheet_names = excel_file.sheet_names
        print(f"📊 Found {len(sheet_names)} sheets: {sheet_names}")

        dataframes = []
        for sheet_name in sheet_names:
            print(f"📋 Processing sheet: {sheet_name}")
            df = pd.read_excel(buffer, sheet_name=sheet_name)
            # Normalize column names to lower snake for mapping
            df.columns = [str(c).strip() for c in df.columns]
            dataframes.append(df)
            print(f"   📈 Sheet '{sheet_name}' has {len(df)} rows")

    records: List[Dict[str, Any]] = []
    total_records = 0

    for i, df in enumerate(dataframes):
        sheet_name = sheet_names[i] if i < len(sheet_names) else f"Sheet_{i+1}"
        print(f"🔄 Processing {sheet_name} with {len(df)} rows...")

        sheet_records = 0
        for _, row in df.iterrows():
            normalized = _normalize_row_from_tabular(row)
            if normalized:
                # Add sheet information to metadata for tracking
                normalized["source_sheet"] = sheet_name
                records.append(normalized)
                sheet_records += 1

        print(
            f"   ✅ Processed {sheet_records} valid records from {sheet_name}")
        total_records += sheet_records

    print(
        f"📊 Total records processed: {total_records} from {len(sheet_names)} sheets"
    )
    return records


def _parse_json_payload(file_bytes: bytes) -> List[Dict[str, Any]]:
    payload = json.loads(file_bytes.decode("utf-8"))
    items = payload if isinstance(payload, list) else [payload]
    records: List[Dict[str, Any]] = []
    for item in items:
        normalized = _normalize_row_from_json(item)
        if normalized:
            records.append(normalized)
    return records


def _parse_products(product_name: str) -> List[str]:
    """Parse product names separated by Chinese commas"""
    if not product_name or pd.isna(product_name):
        return []

    product_str = str(product_name).strip()
    if not product_str:
        return []

    # Split by Chinese comma (，) and other common separators
    separators = ["、", "／"]
    products = []

    for separator in separators:
        if separator in product_str:
            products = [
                p.strip() for p in product_str.split(separator) if p.strip()
            ]
            break

    # If no separators found, return the single product
    if not products:
        products = [product_str]

    return products


def _normalize_row_from_tabular(row: pd.Series) -> Dict[str, Any]:
    get = lambda *keys: next((row.get(k) for k in keys
                              if k in row and not (pd.isna(row.get(k)) or str(
                                  row.get(k)).strip() == "")), None)

    # Handle both English and Mandarin column names
    # English variants
    company_name = get("Company_Name", "company_name", "公司名稱", "公司名称")
    industry_category = get("Industry_category", "Industry",
                            "industry_category", "industry", "產業別", "产业别")
    country = get("country", "Country", "國家", "国家")
    capital_amount = get("Capital_Amour", "capital_amount", "Capital_amount",
                         "資本額", "资本额")
    revenue = get("Revenue", "revenue", "營業額", "营业额")
    cert_docs = get("Company_Certification_Documents",
                    "Company_certification_documents", "cert_docs", "公司認證資料",
                    "公司简介")
    product_name = get("Product_Name", "product_name", "產品名稱", "产品名称")
    main_raw_materials = get("Main_Raw_Materials", "main_raw_materials",
                             "主要原料")
    product_standard = get("Product_Standard", "product_standard",
                           "產品規格(尺寸、精度)", "产品规格(尺寸、精度)")
    technical_advantages = get("Technical_advantages", "technical_advantages",
                               "技術優勢", "技术优势")

    patent_count = get("Patent_Count", "patent_count", "專利數量")

    # Convert types
    def to_int(v):
        try:
            return int(v) if v is not None and str(v).strip() != "" else None
        except Exception:
            return None

    def to_chinese_int(v):
        """Convert Chinese number strings to float values"""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0.0
        return convert_chinese_number_to_int(str(v))

    def to_chinese_int_billions(v):
        """Convert Chinese number strings to float values in billions unit (10^9)"""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0.0
        original_value = str(v)
        result = convert_chinese_number_to_int(original_value,
                                               unit_billions=True)
        print(f"   💰 Converted '{original_value}' to {result} (billions unit)")
        return result

    def to_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        s = str(v).strip().lower()
        return s in ["true", "1", "yes", "y", "是", "有"]

    def to_list(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            # split by commas, semicolons, forward slashes, or Chinese comma
            parts = []
            for separator in ["／", "、"]:
                if separator in v:
                    parts = [
                        p.strip() for p in v.split(separator)
                        if p.strip() != ""
                    ]
                    break
            if not parts:
                parts = [str(v)]
            return parts
        return [str(v)]

    normalized = {
        "company_name":
        str(company_name).strip() if company_name is not None else None,
        "industry_category": to_list(industry_category),  # Vector/array
        "country": str(country).strip() if country is not None else None,
        "capital_amount": to_chinese_int_billions(
            capital_amount
        ),  # Convert Chinese numbers like "20億" to billions unit
        "revenue": to_chinese_int_billions(
            revenue),  # Convert Chinese numbers like "20億" to billions unit
        "company_certification_documents": to_list(cert_docs),  # Vector/array
        "product_name":
        str(product_name).strip() if product_name is not None else None,
        "main_raw_materials": to_list(main_raw_materials),  # Vector/array
        "product_standard": to_list(product_standard),  # Vector/array
        "technical_advantages": to_list(technical_advantages),  # Vector/array
        "patent_count": to_int(patent_count),
    }
    return normalized


def _normalize_row_from_json(item: Dict[str, Any]) -> Dict[str, Any]:
    # Support both English and Chinese field names
    product = item.get("Product") or item.get("產品") or {}

    def get_field(*keys):
        """Get field value from multiple possible key names (English and Chinese)"""
        for key in keys:
            if key in item and item[key] is not None and str(
                    item[key]).strip() != "":
                return item[key]
        return None

    def get_product_field(*keys):
        """Get field value from product object with multiple possible key names"""
        for key in keys:
            if key in product and product[key] is not None and str(
                    product[key]).strip() != "":
                return product[key]
        return None

    def to_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            # split by commas, semicolons, forward slashes, or Chinese comma
            parts = []
            for separator in ["／", "、"]:
                if separator in v:
                    parts = [
                        p.strip() for p in v.split(separator)
                        if p.strip() != ""
                    ]
                    break
            if not parts:
                parts = [str(v)]
            return parts
        return [str(v)]

    def to_chinese_int(v):
        """Convert Chinese number strings to float values"""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0.0
        return convert_chinese_number_to_int(str(v))

    def to_chinese_int_billions(v):
        """Convert Chinese number strings to float values in billions unit (10^9)"""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0.0
        original_value = str(v)
        result = convert_chinese_number_to_int(original_value,
                                               unit_billions=True)
        print(f"   💰 Converted '{original_value}' to {result} (billions unit)")
        return result

    def to_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        s = str(v).strip().lower()
        return s in ["true", "1", "yes", "y", "是", "有"]

    def to_int(v):
        try:
            return int(v) if v is not None and str(v).strip() != "" else 0
        except Exception:
            return 0

    # Handle both English and Chinese field names
    company_name = get_field("Company_Name", "company_name", "公司名稱", "公司名称")
    industry_category = get_field("Industry_category", "Industry",
                                  "industry_category", "industry", "產業別",
                                  "产业别")
    country = get_field("country", "Country", "國家", "国家")
    capital_amount = get_field("Capital_Amount", "capital_amount",
                               "Capital_amount", "資本額", "资本额")
    revenue = get_field("Revenue", "revenue", "營業額", "营业额")
    cert_docs = get_field("Company_Certification_Documents",
                          "Company_certification_documents", "cert_docs",
                          "公司認證資料", "公司认证资料", "公司简介")

    # Product fields with bilingual support
    product_name = get_product_field("Product_Name", "product_name", "產品名稱",
                                     "产品名称")
    main_raw_materials = get_product_field("Main_Raw_Materials",
                                           "main_raw_materials", "主要原料")
    product_standard = get_product_field("Product_Standard",
                                         "product_standard", "產品規格(尺寸、精度)",
                                         "产品规格(尺寸、精度)")
    technical_advantages = get_product_field("Technical_Advantages",
                                             "technical_advantages", "技術優勢",
                                             "技术优势")

    patent_count = get_field("Patent_Count", "patent_count", "專利數量", "专利数量")

    normalized = {
        "company_name":
        str(company_name).strip() if company_name is not None else None,
        "industry_category": to_list(industry_category),  # Array
        "country": str(country).strip() if country is not None else None,
        "capital_amount": to_chinese_int_billions(
            capital_amount),  # Convert Chinese numbers to billions unit
        "revenue": to_chinese_int_billions(
            revenue),  # Convert Chinese numbers to billions unit
        "company_certification_documents": to_list(cert_docs),  # Array
        "product_name":
        str(product_name).strip() if product_name is not None else None,
        "main_raw_materials": to_list(main_raw_materials),  # Array
        "product_standard": to_list(product_standard),  # Array
        "technical_advantages": to_list(technical_advantages),  # Array
        "patent_count":
        to_int(patent_count),  # Convert boolean Patent to count
    }
    return normalized


async def _upsert_company(session: AsyncSession, rec: Dict[str, Any]):
    now = datetime.utcnow().isoformat()
    company_name = rec.get("company_name")
    if not company_name:
        raise HTTPException(status_code=400,
                            detail="Missing company_name in a record")

    result = await session.execute(
        select(Companies).where(Companies.name == company_name))
    existing = result.scalars().first()
    if existing:
        # Handle industry_category as array
        if rec.get("industry_category"):
            existing.industry_category = rec["industry_category"]
        existing.country = rec.get("country") or existing.country
        if rec.get("capital_amount") is not None:
            existing.capital_amount = rec[
                "capital_amount"]  # Float value in billions unit
        if rec.get("revenue") is not None:
            existing.revenue = rec["revenue"]  # Float value in billions unit
        if rec.get("company_certification_documents") is not None:
            existing.company_certification_documents = rec[
                "company_certification_documents"]
        if rec.get("patent_count") is not None:
            existing.patent_count = rec["patent_count"]
        existing.updated_at = now
        return existing, False
    else:
        company = Companies(
            id=str(uuid.uuid4()),
            name=company_name,
            industry_category=rec.get("industry_category")
            or [],  # Array of industries
            country=rec.get("country") or "",
            capital_amount=rec.get("capital_amount")
            or 0.0,  # Float value in billions unit
            revenue=rec.get("revenue") or 0.0,  # Float value in billions unit
            company_certification_documents=rec.get(
                "company_certification_documents") or [],
            patent_count=rec.get("patent_count") or 0,
            created_at=now,
            updated_at=now,
        )
        session.add(company)
        return company, True


async def _upsert_product(session: AsyncSession, company_id: str,
                          rec: Dict[str, Any]):
    now = datetime.utcnow().isoformat()
    product_name = rec.get("product_name")
    if not product_name:
        # Create a shell product if no product fields? Skip instead.
        raise HTTPException(status_code=400,
                            detail="Missing product_name in a record")

    result = await session.execute(
        select(Products).where(Products.company_id == company_id,
                               Products.product_name == product_name))
    existing = result.scalars().first()

    # Handle vector fields properly
    main_raw_materials = rec.get("main_raw_materials") or []
    product_standard = rec.get("product_standard") or []
    technical_advantages = rec.get("technical_advantages") or []

    if existing:
        existing.main_raw_materials = main_raw_materials or existing.main_raw_materials
        existing.product_standard = product_standard or existing.product_standard
        existing.technical_advantages = technical_advantages or existing.technical_advantages
        existing.updated_at = now
        return existing, False
    else:
        product = Products(
            id=str(uuid.uuid4()),
            company_id=company_id,
            product_name=product_name,
            main_raw_materials=main_raw_materials,
            product_standard=product_standard,
            technical_advantages=technical_advantages,
            created_at=now,
            updated_at=now,
        )
        session.add(product)
        return product, True


def _build_metadata(rec: Dict[str, Any]):
    # Prepare pandas series for scoring
    score_series = pd.Series({
        k: rec.get(k)
        for k in [
            "company_name", "industry_category", "country", "capital_amount",
            "revenue", "company_certification_documents", "product_name",
            "main_raw_materials", "product_standard", "technical_advantages",
            "patent_count"
        ]
    })
    data_score = calculate_score(score_series)

    metadata = {
        "company_name": rec.get("company_name"),
        "industry_category": rec.get("industry_category") or [],
        "country": rec.get("country"),
        "product_name": rec.get("product_name"),
        "main_raw_materials": rec.get("main_raw_materials") or [],
        "product_standard": rec.get("product_standard") or [],
        "technical_advantages": rec.get("technical_advantages") or [],
        "patent_count": rec.get("patent_count"),
        "data_score": data_score,
    }
    return metadata, data_score


def _build_metadata_for_industry(rec: Dict[str, Any], industry: str):
    """Build metadata for a specific industry (single industry per vector)"""
    # Prepare pandas series for scoring
    score_series = pd.Series({
        k: rec.get(k)
        for k in [
            "company_name", "country", "capital_amount", "revenue",
            "company_certification_documents", "product_name",
            "main_raw_materials", "product_standard", "technical_advantages",
            "patent_count"
        ]
    })
    # Add the specific industry for scoring
    score_series["industry_category"] = industry
    data_score = calculate_score(score_series)

    metadata = {
        "company_name": rec.get("company_name"),
        "industry_category": industry,  # Single industry string, not array
        "country": rec.get("country"),
        "product_name": rec.get("product_name"),
        "main_raw_materials": rec.get("main_raw_materials") or [],
        "product_standard": rec.get("product_standard") or [],
        "technical_advantages": rec.get("technical_advantages") or [],
        "patent_count": rec.get("patent_count"),
        "data_score": data_score,
    }
    return metadata, data_score


def _build_metadata_for_industry_with_products(rec: Dict[str,
                                                         Any], industry: str,
                                               product_records: List[Any]):
    """Build metadata for a specific industry with all products (industry-based vector)"""
    # Prepare pandas series for scoring
    score_series = pd.Series({
        k: rec.get(k)
        for k in [
            "country", "capital_amount", "revenue",
            "company_certification_documents", "patent_count"
        ]
    })
    # Add the specific industry for scoring
    score_series["industry_category"] = industry
    data_score = calculate_score(score_series)

    # Collect all product information
    products_info = []
    all_main_raw_materials = []
    all_product_standards = []
    all_technical_advantages = []

    for product in product_records:
        products_info.append({
            "product_name":
            product.product_name,
            "main_raw_materials":
            rec.get("main_raw_materials") or [],
            "product_standard":
            rec.get("product_standard") or [],
            "technical_advantages":
            rec.get("technical_advantages") or []
        })
        # Collect all materials, standards, and advantages
        all_main_raw_materials.extend(rec.get("main_raw_materials") or [])
        all_product_standards.extend(rec.get("product_standard") or [])
        all_technical_advantages.extend(rec.get("technical_advantages") or [])

    metadata = {
        "industry_category": industry,  # Single industry string
        "country": rec.get("country"),
        "products": products_info,  # Array of all products for this industry
        "main_raw_materials":
        list(set(all_main_raw_materials)),  # Unique materials
        "product_standard":
        list(set(all_product_standards)),  # Unique standards
        "technical_advantages":
        list(set(all_technical_advantages)),  # Unique advantages
        "patent_count": rec.get("patent_count"),
        "data_score": data_score,
    }
    return metadata, data_score


def _build_embedding_text(metadata: Dict[str, Any]) -> str:
    parts = [
        metadata.get("company_name") or "",
        ", ".join(metadata.get("industry_category")
                  or []),  # Industry categories
        metadata.get("product_name") or "",
        ", ".join(metadata.get("main_raw_materials")
                  or []),  # Main raw materials
        ", ".join(metadata.get("product_standard") or []),  # Product standards
        ", ".join(metadata.get("technical_advantages")
                  or []),  # Technical advantages
    ]
    return " | ".join([str(p) for p in parts if str(p).strip() != ""])


def _build_embedding_text_for_industry(metadata: Dict[str, Any],
                                       industry: str) -> str:
    """Build embedding text for a specific industry (single industry per vector)"""
    parts = [
        metadata.get("company_name") or "",
        industry,  # Single industry string
        metadata.get("product_name") or "",
        ", ".join(metadata.get("main_raw_materials")
                  or []),  # Main raw materials
        ", ".join(metadata.get("product_standard") or []),  # Product standards
        ", ".join(metadata.get("technical_advantages")
                  or []),  # Technical advantages
    ]
    return " | ".join([str(p) for p in parts if str(p).strip() != ""])


def _build_embedding_text_for_industry_with_products(
        metadata: Dict[str,
                       Any], industry: str, product_records: List[Any]) -> str:
    """Build embedding text for a specific industry with all products (industry-based vector)"""
    # Extract product names
    product_names = [p.product_name for p in product_records]

    parts = [
        industry,  # Single industry string
        ", ".join(product_names),  # All product names
        ", ".join(metadata.get("main_raw_materials")
                  or []),  # All raw materials
        ", ".join(metadata.get("product_standard")
                  or []),  # All product standards
        ", ".join(metadata.get("technical_advantages")
                  or []),  # All technical advantages
    ]
    return " | ".join([str(p) for p in parts if str(p).strip() != ""])


async def _create_embedding_async(text: str, model: str,
                                  api_key: str) -> List[float]:
    """Create embeddings using OpenAI client in a background thread.
    The OpenAI Python client methods are synchronous; we offload to a thread.
    """
    # print(f"      🔧 Embedding function called with:")
    # print(f"         - Text length: {len(text)}")
    # print(f"         - Model: {model}")
    # print(f"         - API key present: {bool(api_key)}")

    try:
        client = OpenAI(api_key=api_key)

        # print(f"      ✅ OpenAI client created successfully")

        def _sync_create() -> List[float]:
            # print(f"      🔧 Starting synchronous embedding creation...")
            try:
                resp = client.embeddings.create(model=model, input=text)
                # print(f"      ✅ OpenAI API call successful")
                # print(f"      📊 Response data length: {len(resp.data)}")
                embedding = resp.data[0].embedding
                # print(f"      📊 Embedding dimension: {len(embedding)}")
                return embedding
            except Exception as sync_error:
                # print(
                #     f"      ❌ Synchronous embedding creation failed: {sync_error}"
                # )
                # print(f"      🔧 Sync error type: {type(sync_error).__name__}")
                raise

        # print(f"      🔧 Calling asyncio.to_thread...")
        result = await asyncio.to_thread(_sync_create)
        # print(f"      ✅ Embedding creation completed successfully")
        return result
    except Exception as e:
        print(f"      ❌ Embedding creation failed: {e}")
        print(f"      🔧 Error type: {type(e).__name__}")
        print(f"      🔧 Error details: {str(e)}")
        raise


async def _upsert_vector(session: AsyncSession, company_id: str,
                         product_id: str, embedding: List[float],
                         metadata: Dict[str, Any]):
    now = datetime.utcnow().isoformat()
    # Check if there is an existing vector for this product
    result = await session.execute(
        select(VectorDB).where(VectorDB.product_id == product_id))
    existing = result.scalars().first()
    if existing:
        existing.embedding = embedding
        existing.metadata_json = metadata
        existing.updated_at = now
        return existing, False
    else:
        vector = VectorDB(
            id=str(uuid.uuid4()),
            product_id=product_id,
            company_id=company_id,
            embedding=embedding,
            metadata_json=metadata,
            created_at=now,
            updated_at=now,
        )
        session.add(vector)
        return vector, True


async def _upsert_vector_for_industry(session: AsyncSession, company_id: str,
                                      product_id: str, industry_category: str,
                                      embedding: List[float],
                                      metadata: Dict[str, Any]):
    """Upsert vector for a specific industry (one vector per industry)"""
    now = datetime.utcnow().isoformat()
    # Check if there is an existing vector for this product and industry combination
    result = await session.execute(
        select(VectorDB).where(VectorDB.product_id == product_id,
                               VectorDB.industry_category == industry_category)
    )
    existing = result.scalars().first()
    if existing:
        existing.embedding = embedding
        existing.metadata_json = metadata
        existing.updated_at = now
        return existing, False
    else:
        vector = VectorDB(
            id=str(uuid.uuid4()),
            product_id=product_id,
            company_id=company_id,
            industry_category=industry_category,  # Single industry per vector
            embedding=embedding,
            metadata_json=metadata,
            created_at=now,
            updated_at=now,
        )
        session.add(vector)
        return vector, True


async def _upsert_vector_for_industry_only(session: AsyncSession,
                                           company_id: str,
                                           industry_category: str,
                                           embedding: List[float],
                                           metadata: Dict[str, Any]):
    """Upsert vector for a specific industry only (no specific product)"""
    now = datetime.utcnow().isoformat()
    # Check if there is an existing vector for this company and industry combination
    result = await session.execute(
        select(VectorDB).where(
            VectorDB.company_id == company_id,
            VectorDB.industry_category == industry_category,
            VectorDB.product_id.is_(None)  # No specific product
        ))
    existing = result.scalars().first()
    if existing:
        existing.embedding = embedding
        existing.metadata_json = metadata
        existing.updated_at = now
        return existing, False
    else:
        vector = VectorDB(
            id=str(uuid.uuid4()),
            product_id=None,  # No specific product
            company_id=company_id,
            industry_category=industry_category,  # Single industry per vector
            embedding=embedding,
            metadata_json=metadata,
            created_at=now,
            updated_at=now,
        )
        session.add(vector)
        return vector, True


# # Include the router in the app
# app.include_router(router)

# # Export the app for Vercel
# handler = app
