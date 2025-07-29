import os
import re
from collections import defaultdict
from azure.cosmos import CosmosClient, exceptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
COSMOS_URL = os.getenv("COSMOS_DB_URL")
COSMOS_KEY = os.getenv("COSMOS_DB_KEY")

# Connect to Cosmos DB
cosmos_client = CosmosClient(COSMOS_URL, COSMOS_KEY)
db = cosmos_client.get_database_client("Sponsership")
container = db.get_container_client("companies")

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]+', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

print("📥 Fetching all companies...")
companies = []
for item in container.query_items(
    query="SELECT c.id, c['Company Name'], c['City'] FROM c",
    enable_cross_partition_query=True
):
    companies.append({
        "id": item["id"],
        "original_name": item.get("Company Name", ""),
        "partition_key": item.get("City", "Milwaukee"),
        "normalized_name": normalize_name(item.get("Company Name", ""))
    })

# Group by normalized name
print("🔍 Identifying duplicates...")
name_map = defaultdict(list)
for comp in companies:
    name_map[comp["normalized_name"]].append(comp)

# Safely delete duplicates
total_deleted = 0
for name, records in name_map.items():
    if len(records) > 1:
        keep = records[0]
        to_delete = records[1:]
        for record in to_delete:
            try:
                # Try to read item before delete
                container.read_item(item=record["id"], partition_key=record["partition_key"])
                container.delete_item(item=record["id"], partition_key=record["partition_key"])
                print(f"🗑️ Deleted: {record['original_name']} (id: {record['id']})")
                total_deleted += 1
            except exceptions.CosmosResourceNotFoundError:
                print(f"⚠️ Already deleted or not found: {record['original_name']} (id: {record['id']})")
            except Exception as e:
                print(f"❌ Failed to delete {record['original_name']} (id: {record['id']}) — {e}")

print(f"\n✅ Deduplication complete. Total removed: {total_deleted}")
