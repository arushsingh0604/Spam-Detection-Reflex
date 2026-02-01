import joblib
import os

# --- Robust Path Resolution for Linux ---
# Get the absolute path of the directory where this file sits
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to find the .pkl files in the root folder
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))

MODEL_PATH = os.path.join(root_dir, "spam_classifier_model.pkl")
VEC_PATH = os.path.join(root_dir, "tfidf_vectorizer.pkl")

# Lazy Load Models
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model at {MODEL_PATH}")
    
    MODEL = joblib.load(MODEL_PATH)
    VECTORIZER = joblib.load(VEC_PATH)
except Exception as e:
    print(f"CRITICAL BACKEND ERROR: {e}")
    MODEL = None
    VECTORIZER = None

def analyze_content(text: str):
    """
    Expert-level prediction engine.
    Returns a dictionary of UI-ready attributes.
    """
    if MODEL is None or VECTORIZER is None:
        return {"label": "ENGINE OFFLINE", "confidence": 0.0, "color": "gray"}

    # Vectorize and Predict
    vec = VECTORIZER.transform([text])
    prob = float(MODEL.predict_proba(vec)[0][1])
    
    # 25-Year Expert Tip: Use a threshold of 0.5 for binary classification
    if prob > 0.5:
        return {
            "label": "THREAT DETECTED",
            "confidence": prob,
            "color": "red"
        }
    
    return {
        "label": "VERIFIED SECURE",
        "confidence": 1.0 - prob,
        "color": "emerald"
    }