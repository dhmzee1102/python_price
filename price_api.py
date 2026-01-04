from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union, List, Dict, Any
import joblib
import pandas as pd
import os
import sys

# --- FIX: Ensure imports work regardless of execution directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# ---------------------------------------------------------------

from train_price_model import train_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(current_dir, "price_model.pkl")
artifacts = {}

def load_artifacts():
    global artifacts
    if os.path.exists(MODEL_PATH):
        try:
            artifacts = joblib.load(MODEL_PATH)
            print(f"✅ Loaded {len(artifacts.get('premises', []))} premises and {len(artifacts.get('items', []))} items.")
        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            artifacts = {}
    else:
        print("⚠️ Model file not found. Please trigger training.")

load_artifacts()

# FIX: Allow both int and str to prevent 422 Errors
class PredictionRequest(BaseModel):
    item_code: Union[str, int]
    premise_code: Union[str, int]
    state: str = None
    district: str = None

@app.get("/")
def root():
    return {"message": "Price API is running"}

@app.get("/items")
def get_items():
    """
    Transforms the data to match Flutter's expected JSON format.
    Flutter expects: {'name': ...}
    Database has: {'item': ...}
    """
    raw_items = artifacts.get("items", [])
    transformed_items = []
    
    for i in raw_items:
        transformed_items.append({
            "item_code": i.get("item_code"),
            # FIX: Map 'item' -> 'name' so Flutter displays the name instead of "Unknown"
            "name": i.get("item", "Unknown Item"), 
            "category": i.get("item_category", ""),
            "group": i.get("item_group", "")
        })
        
    return transformed_items

@app.get("/premises")
def get_premises():
    return artifacts.get("premises", [])

@app.post("/predict_price")
def predict_price(request: PredictionRequest):
    if "model" not in artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        model = artifacts["model"]
        
        # 1. Safe integer conversion for lookup
        # (Handles both string "92" and int 92 input from Flutter)
        try:
            req_premise_id = int(request.premise_code)
        except:
            req_premise_id = -1

        # 2. Lookup premise details
        premise_info = next((p for p in artifacts["premises"] if p['premise_code'] == req_premise_id), None)
        
        state = premise_info.get('state', 'Unknown') if premise_info else 'Unknown'
        district = premise_info.get('district', 'Unknown') if premise_info else 'Unknown'
        premise_name = premise_info.get('premise', 'Unknown Premise') if premise_info else 'Unknown Premise'
        
        # 3. Prepare input for model
        # The model was trained with string IDs, so we convert inputs to string here
        now = pd.Timestamp.now()
        input_df = pd.DataFrame({
            'item_code': [str(request.item_code)],
            'month': [now.month],
            'day_of_week': [now.dayofweek],
            'state': [str(state)],
            'district': [str(district)]
        })
        
        # 4. Predict
        prediction = model.predict(input_df)[0]
        
        # 5. Get Item Name for response
        try:
            req_item_id = int(request.item_code)
        except:
            req_item_id = -1
            
        item_info = next((i for i in artifacts['items'] if i['item_code'] == req_item_id), None)
        item_name = item_info['item'] if item_info else "Unknown Item"

        return {
            "item": item_name,
            "premise": premise_name,
            "predicted_price_rm": float(round(prediction, 2)),
            "date": now.strftime("%Y-%m-%d"),
            "note": "AI Estimate"
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Print stack trace to console to help debugging if this happens again
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def trigger_training():
    try:
        result = train_model()
        if result['status'] == 'success':
            load_artifacts()
            return result
        else:
            raise HTTPException(status_code=500, detail=result['message'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))