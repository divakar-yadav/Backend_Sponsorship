# sponsor-backend/app.py

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS, cross_origin
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient, exceptions
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import uuid, os, io, json, datetime
import jwt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://40.76.124.219"]}})

UPLOAD_FOLDER = './temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



BLOB_CONN_STR = os.getenv("AZURE_BLOB_CONN_STR")
BLOB_CONTAINER = os.getenv("AZURE_DATASET_CONTAINER")
EXCEL_FILE_BLOB = os.getenv("AZURE_EXCEL_BLOB_NAME")
MODEL_UPLOAD_CONTAINER = os.getenv("AZURE_MODEL_CONTAINER")

COSMOS_URL = os.getenv("COSMOS_DB_URL")
COSMOS_KEY = os.getenv("COSMOS_DB_KEY")




blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
dataset_container = blob_service.get_container_client(BLOB_CONTAINER)
model_container = blob_service.get_container_client(MODEL_UPLOAD_CONTAINER)

# Cosmos DB Setup

cosmos_client = CosmosClient(COSMOS_URL, COSMOS_KEY)
db = cosmos_client.get_database_client("Sponsership")
training_meta = db.get_container_client("TrainingJobs")
model_meta = db.get_container_client("models")
company_meta = db.get_container_client("companies")
auth_container = db.get_container_client("users")

JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            input_data = request.get_json()
            df = pd.DataFrame(input_data["companies"])
        else:
            file = request.files.get("file")
            if not file:
                return jsonify({'error': 'CSV file is required'}), 400
            df = pd.read_csv(file)

        current_models = list(model_meta.query_items(
            query="SELECT * FROM c WHERE c.status = 'Current'",
            enable_cross_partition_query=True
        ))

        if not current_models:
            return jsonify({"error": "No current model available"}), 404

        model_blob_name = current_models[0]["model_blob_name"]
        blob_client = model_container.get_blob_client(blob=model_blob_name)
        with open("temp_model.pkl", "wb") as f:
            f.write(blob_client.download_blob().readall())

        model, imputer, scaler = load("temp_model.pkl")
        os.remove("temp_model.pkl")

        features = [
            'Annual Revenue in Log', 'Market Valuation in Log', 'Profit Margins', 'Market Share',
            'Industry Ranking', 'Distance', 'University Student Size', 'University Ranking'
        ]

        X_input = scaler.transform(imputer.transform(df[features]))
        probs = model.predict_proba(X_input)[:, 1]

        result = sorted([
            {"company": df.iloc[i].get("Company Name", f"Row {i+1}"), "probability": round(float(probs[i]), 4)}
            for i in range(len(probs))
        ], key=lambda x: x["probability"], reverse=True)

        return jsonify({"ranked_predictions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/upload-data', methods=['POST'])
def upload_data():
    file = request.files['file']
    done_by = request.form.get("done_by", "Unknown User")
    filename = secure_filename(file.filename)

    try:
        new_df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        file.seek(0)
        new_df = pd.read_csv(file, encoding='ISO-8859-1')

    # Save CSV locally
    temp_csv_path = os.path.join(UPLOAD_FOLDER, filename)
    new_df.to_csv(temp_csv_path, index=False)

    # Upload file to Blob Storage
    with open(temp_csv_path, 'rb') as f:
        dataset_container.upload_blob(name=filename, data=f, overwrite=True)

    # Save metadata to Cosmos DB
    dataset_id = str(uuid.uuid4())
    dataset_meta = {
        'id': dataset_id,
        'dataSetId': dataset_id,  # <-- Ensure partition key exists
        'filename': filename,
        'uploaded_at': datetime.datetime.utcnow().isoformat(),
        'num_rows': len(new_df),
        'num_columns': len(new_df.columns),
        'columns': new_df.columns.tolist(),
        'done_by': done_by  # <-- Store the user who uploaded the data
        
    }

    # Ensure container exists
    dataset_meta_container = db.get_container_client("datasets")
    dataset_meta_container.create_item(body=dataset_meta)
    os.remove(temp_csv_path)

    return jsonify({
        "status": "Dataset uploaded successfully.",
        "dataset_id": dataset_id
    })

@app.route('/download-dataset/<dataset_id>', methods=['GET'])
def download_dataset(dataset_id):
    try:
        dataset_meta_container = db.get_container_client("datasets")
        meta = dataset_meta_container.read_item(item=dataset_id, partition_key=dataset_id)
        blob_name = meta.get("filename")

        if not blob_name:
            return jsonify({"error": "No filename associated with this dataset ID"}), 404

        sas_url = generate_blob_sas(
            account_name=blob_service.account_name,
            container_name=BLOB_CONTAINER,
            blob_name=blob_name,
            account_key=blob_service.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        )

        blob_url = f"https://{blob_service.account_name}.blob.core.windows.net/{BLOB_CONTAINER}/{blob_name}?{sas_url}"
        return jsonify({"url": blob_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def train_model_logic(file_stream, filename, dataset_id, model_meta, dataset_container, blob_service, MODEL_UPLOAD_CONTAINER, done_by):
    # Load dataset
    df = pd.read_csv(file_stream) if filename.endswith(".csv") else pd.read_excel(file_stream)

    # Define features and target
    features = [
        'Annual Revenue in Log', 'Market Valuation in Log', 'Profit Margins', 'Market Share',
        'Industry Ranking', 'Distance', 'University Student Size', 'University Ranking'
    ]
    target = 'Sponsored'
    df_clean = df.dropna(subset=[target])
    X_raw = df_clean[features]
    y = df_clean[target].values

    # Preprocessing
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X_raw))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_train, y_train)

    # Evaluation metrics
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]

    # Save model
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    short_uuid = uuid.uuid4().hex[:8]
    model_filename = f"trained_model_{timestamp}_{short_uuid}.pkl"
    model_id = f"{timestamp}_{short_uuid}"
    dump((clf, imputer, scaler), model_filename)

    # Upload model
    model_blob_client = blob_service.get_blob_client(container=MODEL_UPLOAD_CONTAINER, blob=model_filename)
    with open(model_filename, "rb") as f:
        model_blob_client.upload_blob(f, overwrite=True)
    os.remove(model_filename)

    # Archive existing "Current" models
    current_models = list(model_meta.query_items(
        query="SELECT * FROM c WHERE c.status = 'Current'",
        enable_cross_partition_query=True
    ))
    for model in current_models:
        model["status"] = "Archived"
        model_meta.replace_item(item=model["id"], body=model)

    # Save new metadata
    model_meta.create_item({
        "id": model_id,
        "model_id": model_id,
        "model_blob_name": model_filename,
        "done_by": done_by,  # Placeholder, can be updated later
        "created_at": datetime.datetime.utcnow().isoformat(),
        "status": "Current",
        "dataset_id": dataset_id,
        "filename": filename,
        "metrics": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "auc": round(auc, 4),
            "roc_curve": roc_points,
            "confusion_matrix": {
                "truePositive": int(tp),
                "falsePositive": int(fp),
                "trueNegative": int(tn),
                "falseNegative": int(fn)
            }
        }
    })

    return {
        "message": "New model trained and set to 'Current'",
        "model_id": model_id
    }


@app.route('/train-model', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        dataset_id = data.get("dataset_id")
        done_by = data.get("done_by", "Unknown User")
        if not dataset_id:
            return jsonify({"error": "dataset_id is required"}), 400

        dataset_meta_container = db.get_container_client("datasets")
        dataset_meta = dataset_meta_container.read_item(item=dataset_id, partition_key=dataset_id)
        filename = dataset_meta.get("filename")

        blob_client = dataset_container.get_blob_client(blob=filename)
        stream = io.BytesIO()
        blob_data = blob_client.download_blob()
        blob_data.readinto(stream)
        stream.seek(0)

        result = train_model_logic(
            file_stream=stream,
            filename=filename,
            dataset_id=dataset_id,
            model_meta=model_meta,
            dataset_container=dataset_container,
            blob_service=blob_service,
            MODEL_UPLOAD_CONTAINER=MODEL_UPLOAD_CONTAINER,
            done_by=done_by
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/current-model-performance', methods=['GET'])
def get_current_model_performance():
    try:
        query = "SELECT * FROM c WHERE c.status = 'Current'"
        current_models = list(model_meta.query_items(query=query, enable_cross_partition_query=True))
        
        if not current_models:
            return jsonify({"error": "No current model found"}), 404

        model = current_models[0]  # Assuming only one model is Current
        return jsonify(model)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/companies', methods=['GET'])
def get_milwaukee_companies():
    try:
        query = "SELECT * FROM c WHERE c.city = 'Milwaukee'"
        items = list(company_meta.query_items(query=query, enable_cross_partition_query=True))
        return jsonify({"companies": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/training-status', methods=['GET'])
def training_status():
    status = training_meta.read_item(item="latest", partition_key="latest").get('status', 'unknown')
    return jsonify(status=status)

@app.route('/list-models', methods=['GET'])
def list_models_from_db():
    try:
        # Fetch all model documents from the 'models' container
        models = list(model_meta.read_all_items())

        # Return the entire list of model metadata
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/list-training-data', methods=['GET'])
def list_training_data():
    try:
        dataset_meta_container = db.get_container_client("datasets")

        items = list(dataset_meta_container.read_all_items())
        result = []

        for item in items:
            result.append({
                "dataset_id": item.get("dataSetId", item.get("id")),
                "filename": item.get("filename"),
                "uploaded_at": item.get("uploaded_at"),
                "num_rows": item.get("num_rows"),
                "num_columns": item.get("num_columns"),
                "columns": item.get("columns", []),
                "done_by": item.get("done_by", "Unknown User")
            })

        return jsonify({"datasets": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/list-companies', methods=['GET'])
def list_companies():
    try:
        query = "SELECT * FROM c WHERE c.city = 'Milwaukee'"
        items = list(company_meta.query_items(query=query, enable_cross_partition_query=True))
        return jsonify(companies=items)
    except exceptions.CosmosHttpResponseError as e:
        return jsonify(error=str(e)), 500

@app.route('/models-meta-data', methods=['GET'])
def list_model_metadata():
    try:
        items = list(model_meta.read_all_items())
        return jsonify(items)
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/deploy-model', methods=['POST'])
def deploy_model():
    try:
        data = request.get_json()
        model_id = data.get("model")
        done_by = data.get("done_by", "Unknown User")

        if not model_id:
            return jsonify({"error": "Model ID is required"}), 400

        # Archive all current models
        current_models = list(model_meta.query_items(
            query="SELECT * FROM c WHERE c.status = 'Current'",
            enable_cross_partition_query=True
        ))
        for model in current_models:
            model["status"] = "Archived"
            model_meta.replace_item(item=model["id"], body=model)

        # Find and set the selected model as current
        target_models = list(model_meta.query_items(
            query="SELECT * FROM c WHERE c.model_id = @model_id",
            parameters=[{"name": "@model_id", "value": model_id}],
            enable_cross_partition_query=True
        ))

        if not target_models:
            return jsonify({"error": f"Model ID {model_id} not found"}), 404

        selected_model = target_models[0]
        selected_model["status"] = "Current"
        selected_model["deployed_at"] = datetime.datetime.utcnow().isoformat()
        selected_model["done_by"] = done_by  # Update the user who deployed the model
        model_meta.replace_item(item=selected_model["id"], body=selected_model)

        return jsonify({"message": f"Model {model_id} deployed as 'Current'"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/api/signup', methods=['POST'])
@cross_origin()
def signup():
    data = request.json
    name = data.get('name')  # ✅ Make sure frontend sends this
    email = data.get('email')
    password = data.get('password')

    if not name or not email or not password:
        return jsonify({'error': 'All fields required'}), 400

    try:
        query = f"SELECT * FROM c WHERE c.email = '{email}'"
        existing = list(auth_container.query_items(query=query, enable_cross_partition_query=True))

        if existing:
            return jsonify({'error': 'Email already registered'}), 409

        hashed_password = generate_password_hash(password)
        user_id = str(uuid.uuid4())

        # ✅ Store name in the user item
        auth_container.create_item({
            'id': user_id,
            'name': name,  # <-- This is the only field you’re adding to Cosmos
            'email': email,
            'password': hashed_password,
            'created_at': datetime.datetime.utcnow().isoformat()
        })

        return jsonify({'message': 'User registered successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === LOGIN (Cosmos DB) ===
@app.route('/api/login', methods=['POST'])
@cross_origin()
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    try:
        query = f"SELECT * FROM c WHERE c.email = '{email}'"
        results = list(auth_container.query_items(query=query, enable_cross_partition_query=True))
        if not results:
            return jsonify({'error': 'Invalid credentials'}), 401

        user = results[0]
        if not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid credentials'}), 401

        token = jwt.encode({
            'user_id': user['id'],
            'email': user['email'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }, JWT_SECRET, algorithm='HS256')

        # Fix: decode token if it's bytes
        if isinstance(token, bytes):
            token = token.decode('utf-8')

        return jsonify({
            'token': token,
            'user': {
                'name': user['name'],
                'email': user['email']
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
