from azure.cosmos import CosmosClient
import uuid, datetime, re
import random
import os
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
print("✅ Old companies removed.")

# Hardcoded Milwaukee companies with dummy financial data
companies = [
    ("Northwestern Mutual", 15.3, 16.2, 0.18, 2.1, 3),
    ("Harley-Davidson", 13.9, 15.1, 0.14, 1.6, 5),
    ("Fiserv", 15.0, 16.8, 0.20, 2.4, 2),
    ("ManpowerGroup", 14.5, 15.6, 0.12, 1.2, 7),
    ("Rockwell Automation", 14.2, 15.5, 0.17, 2.0, 6),
    ("Johnson Controls", 15.1, 16.3, 0.19, 2.5, 1),
    ("Brady Corporation", 13.1, 14.0, 0.13, 0.9, 10),
    ("Badger Meter", 12.7, 13.6, 0.11, 0.8, 12),
    ("REV Group", 13.5, 14.2, 0.10, 1.1, 11),
    ("Direct Supply", 13.0, 13.8, 0.09, 0.7, 13),
    ("MGIC Investment", 13.8, 14.5, 0.15, 1.3, 9),
    ("WEC Energy Group", 14.6, 15.7, 0.21, 2.2, 4),
    ("Marcus Corporation", 12.6, 13.5, 0.08, 0.6, 14),
    ("A. O. Smith Corporation", 13.9, 14.7, 0.16, 1.4, 8),
    ("Zurn Elkay Water Solutions", 13.2, 14.1, 0.12, 1.0, 15),
    ("GE HealthCare", 15.2, 16.4, 0.22, 2.6, 16),
    ("Sensient Technologies", 13.4, 14.3, 0.10, 0.9, 17),
    ("Joy Global (Komatsu)", 14.1, 15.0, 0.13, 1.5, 18),
    ("Astronautics Corporation", 12.8, 13.9, 0.07, 0.5, 19),
    ("Pentair", 14.0, 15.2, 0.18, 1.9, 20),
]

# Ingest real company data
print("📥 Ingesting Milwaukee companies...")
for name, rev, val, margin, share, rank in companies:
    company_doc = {
        "id": str(uuid.uuid4()),
        "Company Name": name,
        "Annual Revenue in Log": rev,
        "Market Valuation in Log": val,
        "Profit Margins": margin,
        "Market Share": share,
        "Industry Ranking": rank,
        "Distance": round(random.uniform(1.0, 10.0), 2),
        "city": "Milwaukee",
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    company_meta.create_item(company_doc)
print("✅ Ingestion complete.")
