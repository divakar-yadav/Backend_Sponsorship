from azure.cosmos import CosmosClient, PartitionKey, exceptions
import uuid
import datetime
import os

# Cosmos DB config
COSMOS_URL = os.getenv("COSMOS_DB_URL")
COSMOS_KEY = os.getenv("COSMOS_DB_KEY")
DATABASE_NAME = "Sponsership"  # ✅ correct spelling
CONTAINER_NAME = "models"

# Initialize Cosmos DB client
client = CosmosClient(COSMOS_URL, credential=COSMOS_KEY)

# Get database and container clients
db = client.get_database_client(DATABASE_NAME)
container = db.get_container_client(CONTAINER_NAME)

# Debug: Show actual function to ensure SDK is correct
print("create_item method reference:", container.create_item)

# Sample documents
docs = [
    {
        "id": str(uuid.uuid4()),
        "modelName": "logistic-reg-v1",
        "status": "Current",
        "createdAt": datetime.datetime.utcnow().isoformat()
    },
    {
        "id": str(uuid.uuid4()),
        "modelName": "logistic-reg-v2",
        "status": "Archived",
        "createdAt": datetime.datetime.utcnow().isoformat()
    },
    {
        "id": str(uuid.uuid4()),
        "modelName": "logistic-reg-v3",
        "status": "Testing",
        "createdAt": datetime.datetime.utcnow().isoformat()
    },
    {
        "id": str(uuid.uuid4()),
        "modelName": "ensemble-v1",
        "status": "Current",
        "createdAt": datetime.datetime.utcnow().isoformat()
    },
    {
        "id": str(uuid.uuid4()),
        "modelName": "tree-boost-v1",
        "status": "Archived",
        "createdAt": datetime.datetime.utcnow().isoformat()
    }
]

# Insert documents with correct partition key
for doc in docs:
    try:
        container.create_item(body=doc)
        print(f"✅ Inserted model: {doc['modelName']}")
    except exceptions.CosmosHttpResponseError as e:
        print(f"❌ Cosmos SDK Error inserting {doc['modelName']}: {e.message}")
    except Exception as e:
        print(f"❌ General Error inserting {doc['modelName']}: {e}")
