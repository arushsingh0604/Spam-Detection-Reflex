<<<<<<< HEAD
import reflex as rx
import joblib
import time
import os

# --- 1. Load ML Assets (Robust Pathing) ---
# We try multiple path options to ensure Windows finds the files
MODEL_PATH ="/home/ubuntu/Spam-Detection-Reflex/spam_classifier_model.pkl"
VEC_PATH ="/home/ubuntu/Spam-Detection-Reflex/tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH):
    # Fallback to absolute path if relative fails
    MODEL_PATH = "/home/ubuntu/Spam-Detection-Reflex/spam_classifier_model.pkl"
    VEC_PATH = "/home/ubuntu/Spam-Detection-Reflex/tfidf_vectorizer.pkl"



try:
    MODEL = joblib.load(MODEL_PATH)
    VECTORIZER = joblib.load(VEC_PATH)
except Exception as e:
    print(f"CRITICAL ERROR: Could not find models. {e}")

class State(rx.State):
    """The app state with explicit setters to avoid deprecation warnings."""
    message: str = ""
    prediction: str = "System Ready"
    confidence: float = 0.0
    is_loading: bool = False
    result_color: str = "gray"

    def set_message(self, text: str):
        """Explicit setter for the message variable."""
        self.message = text

    def run_analysis(self):
        if not self.message.strip():
            return rx.window_alert("Please enter a message.")
        
        self.is_loading = True
        yield 
        
        time.sleep(0.8)
        
        vec = VECTORIZER.transform([self.message])
        pred = MODEL.predict(vec)[0]
        prob = MODEL.predict_proba(vec)[0][1]
        
        if pred == 1:
            self.prediction = "SPAM DETECTED"
            self.confidence = float(prob)
            self.result_color = "red"
        else:
            self.prediction = "VERIFIED SECURE"
            self.confidence = float(1 - prob)
            self.result_color = "green"
            
        self.is_loading = False

def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.heading("SENTINELLA AI", size="9", color_scheme="indigo", weight="bold"),
            rx.text("Enterprise Intelligence for Communication Security", color_scheme="gray"),
            
            rx.box(
                rx.vstack(
                    rx.text("Neural Pattern Analysis", size="4", weight="medium"),
                    rx.text_area(
                        placeholder="Paste suspicious content here...",
                        on_change=State.set_message,
                        width="100%",
                        height="180px",
                        variant="soft",
                        border_radius="12px",
                    ),
                    rx.button(
                        "Execute Security Scan",
                        on_click=State.run_analysis,
                        is_loading=State.is_loading,
                        width="100%",
                        size="3",
                        color_scheme="indigo",
                    ),
                    spacing="4",
                ),
                padding="2.5em",
                border_radius="24px",
                bg="rgba(255, 255, 255, 0.03)", 
                backdrop_filter="blur(15px)", 
                border="1px solid rgba(255, 255, 255, 0.08)",
                width="100%",
            ),

            rx.cond(
                State.prediction != "System Ready",
                rx.vstack(
                    rx.divider(margin_y="1.5em", opacity="0.2"),
                    rx.heading(State.prediction, size="8", color=State.result_color),
                    # Fix 1: Cast to int using .to(int) for the progress bar
                    rx.progress(
                        value=(State.confidence * 100).to(int), 
                        width="100%", 
                        color_scheme=State.result_color,
                    ),
                    # Fix 2: Use .to_string() for formatting in the UI
                    rx.text(
                        "Confidence: " + (State.confidence * 100).to_string() + "%", 
                        weight="light"
                    ),
                    width="100%",
                    align="center",
                    spacing="3",
                ),
            ),
            spacing="7",
            width="600px",
        ),
        height="100vh",
        background="radial-gradient(circle at top right, #0F172A, #020617)",
    )

app = rx.App(theme=rx.theme(appearance="dark", accent_color="indigo"))
=======
import reflex as rx
import joblib
import time
import os

# --- 1. Load ML Assets (Robust Pathing) ---
# We try multiple path options to ensure Windows finds the files
MODEL_PATH = "../../spam_classifier_model.pkl"
VEC_PATH = "../../tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH):
    # Fallback to absolute path if relative fails
    MODEL_PATH = "C:/Spam detection folder/spam_classifier_model.pkl"
    VEC_PATH = "C:/Spam detection folder/tfidf_vectorizer.pkl"

try:
    MODEL = joblib.load(MODEL_PATH)
    VECTORIZER = joblib.load(VEC_PATH)
except Exception as e:
    print(f"CRITICAL ERROR: Could not find models. {e}")

class State(rx.State):
    """The app state with explicit setters to avoid deprecation warnings."""
    message: str = ""
    prediction: str = "System Ready"
    confidence: float = 0.0
    is_loading: bool = False
    result_color: str = "gray"

    def set_message(self, text: str):
        """Explicit setter for the message variable."""
        self.message = text

    def run_analysis(self):
        if not self.message.strip():
            return rx.window_alert("Please enter a message.")
        
        self.is_loading = True
        yield 
        
        time.sleep(0.8)
        
        vec = VECTORIZER.transform([self.message])
        pred = MODEL.predict(vec)[0]
        prob = MODEL.predict_proba(vec)[0][1]
        
        if pred == 1:
            self.prediction = "SPAM DETECTED"
            self.confidence = float(prob)
            self.result_color = "red"
        else:
            self.prediction = "VERIFIED SECURE"
            self.confidence = float(1 - prob)
            self.result_color = "green"
            
        self.is_loading = False

def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.heading("SENTINELLA AI", size="9", color_scheme="indigo", weight="bold"),
            rx.text("Enterprise Intelligence for Communication Security", color_scheme="gray"),
            
            rx.box(
                rx.vstack(
                    rx.text("Neural Pattern Analysis", size="4", weight="medium"),
                    rx.text_area(
                        placeholder="Paste suspicious content here...",
                        on_change=State.set_message,
                        width="100%",
                        height="180px",
                        variant="soft",
                        border_radius="12px",
                    ),
                    rx.button(
                        "Execute Security Scan",
                        on_click=State.run_analysis,
                        is_loading=State.is_loading,
                        width="100%",
                        size="3",
                        color_scheme="indigo",
                    ),
                    spacing="4",
                ),
                padding="2.5em",
                border_radius="24px",
                bg="rgba(255, 255, 255, 0.03)", 
                backdrop_filter="blur(15px)", 
                border="1px solid rgba(255, 255, 255, 0.08)",
                width="100%",
            ),

            rx.cond(
                State.prediction != "System Ready",
                rx.vstack(
                    rx.divider(margin_y="1.5em", opacity="0.2"),
                    rx.heading(State.prediction, size="8", color=State.result_color),
                    # Fix 1: Cast to int using .to(int) for the progress bar
                    rx.progress(
                        value=(State.confidence * 100).to(int), 
                        width="100%", 
                        color_scheme=State.result_color,
                    ),
                    # Fix 2: Use .to_string() for formatting in the UI
                    rx.text(
                        "Confidence: " + (State.confidence * 100).to_string() + "%", 
                        weight="light"
                    ),
                    width="100%",
                    align="center",
                    spacing="3",
                ),
            ),
            spacing="7",
            width="600px",
        ),
        height="100vh",
        background="radial-gradient(circle at top right, #0F172A, #020617)",
    )

app = rx.App(theme=rx.theme(appearance="dark", accent_color="indigo"))
>>>>>>> ae3abbbc2e41e8ddf09d614a1558e0a879b3aacb
app.add_page(index, title="Sentinella AI")