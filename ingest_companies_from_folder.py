from azure.cosmos import CosmosClient
import uuid
import datetime
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to Cosmos DB
COSMOS_URL = os.getenv("COSMOS_DB_URL")
COSMOS_KEY = os.getenv("COSMOS_DB_KEY")
cosmos_client = CosmosClient(COSMOS_URL, COSMOS_KEY)
db = cosmos_client.get_database_client("Sponsership")
company_meta = db.get_container_client("companies")

# Delete companies that start with "Company "
print("🧹 Deleting old test companies...")
for item in company_meta.query_items(
    query="SELECT * FROM c WHERE STARTSWITH(c['Company Name'], 'Company ')",
    enable_cross_partition_query=True
):
    company_meta.delete_item(item=item['id'], partition_key='Milwaukee')
print("✅ Old test companies removed.")

# Directory with chunked output CSVs
CHUNKS_DIR = "clean_data_chunks"

# Read and ingest each CSV file from chunk folder
print("📥 Ingesting companies from CSV chunks...")
chunk_files = sorted([
    os.path.join(CHUNKS_DIR, f)
    for f in os.listdir(CHUNKS_DIR)
    if f.endswith(".csv")
])

total_ingested = 0

for csv_file in chunk_files:
    print(f"➡️ Processing: {csv_file}")
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        company_doc = {
            "id": str(uuid.uuid4()),
            "Company Name": row["company_name"],
            "Stock Symbol": row.get("stock_symbol", ""),
            "Tagline": row.get("tagline", ""),
            "Annual Revenue in Log": float(row.get("annual_revenue_log", 0)),
            "Market Valuation in Log": float(row.get("market_valuation_log", 0)),
            "Profit Margins": float(row.get("profit_margins", 0)),
            "Market Share": float(row.get("market_share", 0)),
            "Industry Ranking": int(row.get("industry_ranking", 0)),
            "Distance": float(row.get("distance", 0)),
            "Known Point of Contact": row.get("known_point_of_contact", ""),
            "City": row.get("city", ""),
            "Created At": row.get("created_at", datetime.datetime.utcnow().isoformat()),

            # Additional fields
            "annual_revenue": row.get("annual_revenue", 0),
            "employee_count": row.get("employee_count", 0),
            "mission_statement": row.get("mission_statement", ""),
            "headquarters_location": row.get("headquarters_location", ""),
            "key_contacts": row.get("key_contacts", ""),
            "predicted_shared_values": row.get("predicted_shared_values", ""),
            "early_stage_focus": row.get("early_stage_focus", ""),
            "project_ideation": row.get("project_ideation", ""),
            "existing_coe_projects": row.get("existing_coe_projects", ""),
            "key_focus_areas": row.get("key_focus_areas", ""),
            "assumptions": row.get("assumptions", ""),
            "dependencies": row.get("dependencies", ""),
            "past_higher_ed_giving": row.get("past_higher_ed_giving", "")
        }

        company_meta.create_item(company_doc)
        total_ingested += 1

print(f"✅ Ingestion complete. Total companies ingested: {total_ingested}")
