from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import shap
import io

app = FastAPI(title="Bank Telemarketing XAI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Machine Learning Artifacts...")
# We load the preprocessor, model, and feature names.
# (Ignore the Scikit-learn/XGBoost warnings in the console, they are non-fatal version notices).
preprocessor = joblib.load('preprocessor.joblib')
model = joblib.load('tuned_xgb_model.joblib')
feature_names = joblib.load('feature_names.joblib')

print("Initializing SHAP Explainer dynamically to prevent Numba crashes...")
# Rebuild the explainer natively here instead of loading a fragile pickle file
# --- XGBoost 3.1+ SHAP Bug Fix ---
import builtins
_original_float = builtins.float

def _patched_float(val):
    # If the value is a string with brackets, strip them out
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        val = val[1:-1]
    return _original_float(val)

# Apply the patch, build the explainer, then restore normal behavior
builtins.float = _patched_float
explainer = shap.TreeExplainer(model)
builtins.float = _original_float
# ---------------------------------
print("API is ready.")

@app.get("/")
def health_check():
    return {"status": "API is active and model is loaded."}

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        # 1. Read CSV into Pandas DataFrame
        contents = await file.read()
        raw_df = pd.read_csv(io.BytesIO(contents))
        
        # Keep a copy of raw data to return to the frontend
        response_data = raw_df.copy()
        
        # 2. Preprocess the raw data
        processed_data = preprocessor.transform(raw_df)
        X_infer = pd.DataFrame(processed_data, columns=feature_names)
        
        # 3. Predict Probabilities
        probabilities = model.predict_proba(X_infer)[:, 1]
        
        # 4. Generate SHAP Values for Explainability
        shap_values = explainer(X_infer).values
        
        # 5. Extract top drivers for each customer
        batch_results = []
        for i in range(len(raw_df)):
            customer_shap = shap_values[i]
            
            # Map SHAP values to feature names and sort by absolute impact
            feature_impacts = list(zip(feature_names, customer_shap))
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Separate into top positive and negative drivers
            top_positive = [{"feature": f, "impact": float(v)} for f, v in feature_impacts if v > 0][:3]
            top_negative = [{"feature": f, "impact": float(v)} for f, v in feature_impacts if v < 0][:3]
            
            batch_results.append({
                "row_id": i,
                "subscription_probability": float(probabilities[i]),
                "prediction": "Yes" if probabilities[i] > 0.5 else "No",
                "drivers": {
                    "positive": top_positive,
                    "negative": top_negative
                },
                "raw_data": raw_df.iloc[i].fillna("").to_dict()
            })
            
        return {"total_processed": len(raw_df), "results": batch_results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))