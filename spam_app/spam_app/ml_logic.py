import joblib
import os


# Resolve paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))

MODEL_PATH = os.path.join(root_dir, "spam_classifier_model.pkl")
VEC_PATH = os.path.join(root_dir, "tfidf_vectorizer.pkl")


# Safe load
try:
    MODEL = joblib.load(MODEL_PATH)
    VECTORIZER = joblib.load(VEC_PATH)
except Exception as e:
    print(f"MODEL LOAD ERROR: {e}")
    MODEL, VECTORIZER = None, None


def analyze_content(text: str):
    """Analyze message and return UI-ready spam result."""

    if MODEL is None or VECTORIZER is None:
        return {
            "label": "ENGINE OFFLINE",
            "confidence": 0.0,
            "color": "gray",
            "explanation": "Spam detection engine is unavailable.",
        }

    vec = VECTORIZER.transform([text])
    spam_prob = float(MODEL.predict_proba(vec)[0][1])

    if spam_prob >= 0.6:
        return {
            "label": "SPAM DETECTED",
            "confidence": spam_prob,
            "color": "red",
            "explanation": "Message contains patterns commonly found in spam.",
        }

    return {
        "label": "NOT SPAM",
        "confidence": 1.0 - spam_prob,
        "color": "emerald",
        "explanation": "Message appears legitimate and safe.",
    }
