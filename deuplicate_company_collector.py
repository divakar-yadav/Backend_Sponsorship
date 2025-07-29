import os
import re
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from azure.cosmos import CosmosClient

# Load environment variables
load_dotenv()
COSMOS_URL = os.getenv("COSMOS_DB_URL")
COSMOS_KEY = os.getenv("COSMOS_DB_KEY")
cosmos_client = CosmosClient(COSMOS_URL, COSMOS_KEY)

# Connect to DB
db = cosmos_client.get_database_client("Sponsership")
container = db.get_container_client("companies")

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    # Lowercase, remove non-alphanumerics, collapse whitespace
    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# Collect all companies
print("📥 Fetching all companies from DB...")
companies = []
for item in container.query_items(
    query="SELECT c.id, c['Company Name'] FROM c",
    enable_cross_partition_query=True
):
    companies.append({
        "id": item["id"],
        "original_name": item.get("Company Name", ""),
        "normalized_name": normalize_name(item.get("Company Name", ""))
    })

# Detect duplicates
print("🔍 Checking for duplicates...")
name_map = defaultdict(list)
for company in companies:
    name_map[company["normalized_name"]].append(company)

# Prepare output
duplicates = []
for normalized, items in name_map.items():
    if len(items) > 1:
        for item in items:
            duplicates.append({
                "id": item["id"],
                "original_name": item["original_name"],
                "normalized_name": normalized
            })

# Export to CSV
output_df = pd.DataFrame(duplicates)
output_file = "duplicate_companies_detected.csv"
output_df.to_csv(output_file, index=False)
print(f"✅ Done. Found {len(duplicates)} entries with duplicate names.")
print(f"📄 Output written to: {output_file}")
