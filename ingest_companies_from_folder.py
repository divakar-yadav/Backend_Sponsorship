import uuid
import datetime
import pandas as pd
import os
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

# Utility functions
def parse_list_field(value):
    if pd.isna(value):
        return []
    return [v.strip() for v in str(value).split(';') if v.strip()]

def safe_float(value):
    try:
        return float(value)
    except:
        return None

def safe_int(value):
    try:
        return int(float(value))
    except:
        return None

def safe_str(value):
    return str(value) if pd.notna(value) else None

# Directory containing all chunked CSVs
CHUNKS_DIR = "clean_data_chunks"

# Delete old test companies
print("🧹 Deleting old test companies...")
for item in company_meta.query_items(
    query="SELECT * FROM c WHERE STARTSWITH(c['Company Name'], 'Company ')",
    enable_cross_partition_query=True
):
    company_meta.delete_item(item=item['id'], partition_key='Milwaukee')
print("✅ Old test companies removed.")

# Gather all CSV files
chunk_files = sorted([
    os.path.join(CHUNKS_DIR, f)
    for f in os.listdir(CHUNKS_DIR)
    if f.endswith(".csv")
])

# Ingest all files
total_ingested = 0

for csv_file in chunk_files:
    print(f"📁 Processing: {csv_file}")
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        company_doc = {
            "id": str(uuid.uuid4()),
            "Company Name": safe_str(row.get("company_name")),
            "Stock Symbol": safe_str(row.get("stock_symbol")),
            "Tagline": safe_str(row.get("tagline")),
            "Annual Revenue in Log": safe_float(row.get("annual_revenue_log")),
            "Market Valuation in Log": safe_float(row.get("market_valuation_log")),
            "Profit Margins": safe_float(row.get("profit_margins")),
            "Market Share": safe_float(row.get("market_share")),
            "Industry Ranking": safe_int(row.get("industry_ranking")),
            "Distance": safe_float(row.get("distance")),
            "Known Point of Contact": safe_str(row.get("known_point_of_contact")),
            "City": safe_str(row.get("city")),
            "Created At": safe_str(row.get("created_at")) or datetime.datetime.now(datetime.timezone.utc).isoformat(),

            # Optional / enriched fields
            "annual_revenue": safe_float(row.get("annual_revenue")),
            "employee_count": safe_int(row.get("employee_count")),
            "mission_statement": safe_str(row.get("mission_statement")),
            "headquarters_location": safe_str(row.get("headquarters_location")),
            "key_contacts": parse_list_field(row.get("key_contacts")),
            "predicted_shared_values": parse_list_field(row.get("predicted_shared_values")),
            "early_stage_focus": parse_list_field(row.get("early_stage_focus")),
            "project_ideation": safe_str(row.get("project_ideation")),
            "existing_coe_projects": safe_str(row.get("existing_coe_projects")),
            "key_focus_areas": safe_str(row.get("key_focus_areas")),
            "assumptions": safe_str(row.get("assumptions")),
            "dependencies": safe_str(row.get("dependencies")),
            "past_higher_ed_giving": safe_str(row.get("past_higher_ed_giving"))
        }

        try:
            company_meta.create_item(company_doc)
            total_ingested += 1
        except Exception as e:
            print(f"❌ Failed to ingest: {company_doc['Company Name']} — {e}")

print(f"✅ Ingestion complete. Total companies ingested: {total_ingested}")
