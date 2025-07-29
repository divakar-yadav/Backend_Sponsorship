import uuid
import datetime
import pandas as pd
import os
import math
from dotenv import load_dotenv
from azure.cosmos import CosmosClient

# Load environment variables
load_dotenv()

# Connect to Cosmos DB
COSMOS_URL = os.getenv("COSMOS_DB_URL")
COSMOS_KEY = os.getenv("COSMOS_DB_KEY")
cosmos_client = CosmosClient(COSMOS_URL, COSMOS_KEY)
db = cosmos_client.get_database_client("Sponsership")
company_meta = db.get_container_client("companies")

# ---- Safe Parsing Utilities ----
def parse_list_field(value):
    try:
        if pd.isna(value) or not str(value).strip():
            return []
        return [v.strip() for v in str(value).split(';') if v.strip()]
    except:
        return []

def safe_float(value):
    try:
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    except:
        return None

def safe_int(value):
    try:
        return int(float(value))
    except:
        return None

def safe_str(value):
    try:
        s = str(value).strip()
        return s.encode("utf-8", "ignore").decode("utf-8") if s else None
    except:
        return None

# ---- Clean Old Test Data ----
print("🧹 Deleting old test companies...")
for item in company_meta.query_items(
    query="SELECT * FROM c WHERE STARTSWITH(c['Company Name'], 'Company ')",
    enable_cross_partition_query=True
):
    company_meta.delete_item(item=item['id'], partition_key='Milwaukee')
print("✅ Old test companies removed.")

# ---- Load CSV ----
CSV_FILE = "company_prospects_chunk_0014.csv"
df = pd.read_csv(CSV_FILE)

# ---- Ingest Rows ----
total_ingested = 0
for _, row in df.iterrows():
    company_doc = {
        "id": str(uuid.uuid4()),
        "Company Name": safe_str(row.get("Company Name")),
        "Stock Symbol": safe_str(row.get("Stock Symbol")),
        "Tagline": safe_str(row.get("Tagline")),
        "Annual Revenue in Log": safe_float(row.get("Annual Revenue in Log")),
        "Market Valuation in Log": safe_float(row.get("Market Valuation in Log")),
        "Profit Margins": safe_float(row.get("Profit Margins")),
        "Market Share": safe_float(row.get("Market Share")),
        "Industry Ranking": safe_int(row.get("Industry Ranking")),
        "Distance": safe_float(row.get("Distance")),
        "Known Point of Contact": safe_str(row.get("Known Point of Contact")),
        "City": safe_str(row.get("City")),
        "Created At": safe_str(row.get("Created At")) or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "annual_revenue": safe_float(row.get("annual_revenue")),
        "employee_count": safe_int(row.get("employee_count")),
        "mission_statement": safe_str(row.get("mission_statement")),
        "headquarters_location": safe_str(row.get("headquarters_location")),
        "key_contacts": parse_list_field(row.get("key_contacts")),
        "predicted_shared_values": parse_list_field(row.get("predicted_shared_values")),
        "early_stage_focus": parse_list_field(row.get("early_stage_focus")),
        "project_ideation": safe_str(row.get("project_ideation")),
        "existing_coe_projects": parse_list_field(row.get("existing_coe_projects")),
        "key_focus_areas": parse_list_field(row.get("key_focus_areas")),
        "assumptions": parse_list_field(row.get("assumptions")),
        "dependencies": parse_list_field(row.get("dependencies")),
        "past_higher_ed_giving": parse_list_field(row.get("past_higher_ed_giving")),
    }

    company_doc = {k: v for k, v in company_doc.items() if v is not None}

    try:
        company_meta.create_item(company_doc)
        total_ingested += 1
    except Exception as e:
        print(f"❌ Failed to ingest: {company_doc.get('Company Name')} — {e}")

print(f"✅ Ingestion complete. Total companies ingested: {total_ingested}")
