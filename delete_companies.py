from azure.cosmos import CosmosClient
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

print("🗑️  Starting deletion of all companies...")

# Get all companies
all_companies = list(company_meta.read_all_items())
total_companies = len(all_companies)

if total_companies == 0:
    print("✅ No companies found to delete.")
else:
    print(f"📊 Found {total_companies} companies to delete.")
    
    # Delete each company
    deleted_count = 0
    failed_count = 0
    
    for company in all_companies:
        company_id = company['id']
        company_name = company.get('Company Name', 'Unknown Company')
        company_city = company.get('City', 'Unknown')
        
        print(f"\n🗑️  Attempting to delete: {company_name} (ID: {company_id}, City: {company_city})")
        
        try:
            # Use the actual city as partition key
            company_meta.delete_item(
                item=company_id, 
                partition_key=company_city
            )
            deleted_count += 1
            print(f"   ✅ Successfully deleted: {company_name}")
            
        except Exception as e:
            failed_count += 1
            print(f"   ❌ Failed to delete {company_name}: {str(e)[:100]}...")
    
    print(f"\n📊 Deletion Summary:")
    print(f"   ✅ Successfully deleted: {deleted_count}")
    print(f"   ❌ Failed to delete: {failed_count}")
    print(f"   📈 Total processed: {total_companies}")

print("🎉 Company deletion process completed!")

# Verify deletion
print("\n🔍 Verifying deletion...")
remaining_companies = list(company_meta.read_all_items())
print(f"📊 Remaining companies in table: {len(remaining_companies)}")

if len(remaining_companies) > 0:
    print("⚠️  Some companies still exist. Here are the remaining companies:")
    for company in remaining_companies:
        print(f"   - {company.get('Company Name', 'Unknown')} (ID: {company.get('id', 'Unknown')}, City: {company.get('City', 'Unknown')})")
else:
    print("✅ All companies successfully deleted!") 