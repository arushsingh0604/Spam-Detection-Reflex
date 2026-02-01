<<<<<<< HEAD
import streamlit as st
import joblib
import time
import logging

# -----------------------------
# 1. Page Configuration & Custom UI
# -----------------------------
st.set_page_config(
    page_title="Sentinella | Spam AI",
    layout="wide" # Wide layout looks more like a modern dashboard
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #d1d5db;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        background-color: #4F46E5;
        color: white;
        height: 3em;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 2. Optimized Model Loading (Caching)
# -----------------------------
@st.cache_resource
def load_assets():
    # Adding a small sleep to simulate heavy loading for the splash effect
    model = joblib.load("spam_classifier_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

with st.spinner("Initializing Intelligent Core..."):
    model, vectorizer = load_assets()

# -----------------------------
# 3. Sidebar (Meta-data & Controls)
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3241/3241516.png", width=100)
    st.title("Sentinella AI")
    st.markdown("---")
    st.info("**System Status:** Operational ")
    st.write("This enterprise-grade system utilizes Natural Language Processing to filter malicious communication.")
    
    if st.checkbox("Show Technical Logs"):
        st.caption("Latest Log Entry:")
        try:
            with open("prediction_logs.txt", "r") as f:
                last_line = f.readlines()[-1]
                st.code(last_line)
        except:
            st.write("No logs available yet.")

# -----------------------------
# 4. Main Interface
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.title(" Spam Detection System")
    st.subheader("Neural Text Analysis")
    
    user_message = st.text_area(
        "Analyze Communication:",
        placeholder="Paste the message content here...",
        height=200
    )
    
    predict_btn = st.button("Run Security Scan")

with col2:
    st.markdown("### System Metrics")
    st.write("Real-time classification confidence and latency.")
    placeholder = st.empty() # Used for updating result cards

# -----------------------------
# 5. Prediction Logic
# -----------------------------
if predict_btn:
    if not user_message.strip():
        st.warning(" Action Required: Please input text to analyze.")
    else:
        with st.status("Analyzing message patterns...", expanded=True) as status:
            start_time = time.time()
            
            # NLP Processing
            st.write("Vectorizing input...")
            message_vector = vectorizer.transform([user_message])
            
            st.write("Calculating probability...")
            prediction = model.predict(message_vector)[0]
            probability = model.predict_proba(message_vector)[0][1]
            
            latency = time.time() - start_time
            time.sleep(0.5) # Slight delay for UX
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # UI Results
        with col2:
            if prediction == 1:
                st.error("### SPAM DETECTED")
                st.metric("Risk Level", f"{probability:.1%}", delta="High Risk", delta_color="inverse")
            else:
                st.success("### VERIFIED HAM")
                st.metric("Trust Score", f"{1-probability:.1%}", delta="Safe")
            
            st.divider()
            st.caption(f"Latency: {latency:.4f}s | Engine: Scikit-learn")

        # Log prediction
        logging.basicConfig(filename="prediction_logs.txt", level=logging.INFO, format="%(asctime)s | %(message)s")
        logging.info(f"Msg: {user_message[:30]}... | Pred: {prediction} | Prob: {probability:.4f}")

st.markdown("---")
=======
import streamlit as st
import joblib
import time
import logging

# -----------------------------
# 1. Page Configuration & Custom UI
# -----------------------------
st.set_page_config(
    page_title="Sentinella | Spam AI",
    layout="wide" # Wide layout looks more like a modern dashboard
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #d1d5db;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        background-color: #4F46E5;
        color: white;
        height: 3em;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 2. Optimized Model Loading (Caching)
# -----------------------------
@st.cache_resource
def load_assets():
    # Adding a small sleep to simulate heavy loading for the splash effect
    model = joblib.load("spam_classifier_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

with st.spinner("Initializing Intelligent Core..."):
    model, vectorizer = load_assets()

# -----------------------------
# 3. Sidebar (Meta-data & Controls)
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3241/3241516.png", width=100)
    st.title("Sentinella AI")
    st.markdown("---")
    st.info("**System Status:** Operational ")
    st.write("This enterprise-grade system utilizes Natural Language Processing to filter malicious communication.")
    
    if st.checkbox("Show Technical Logs"):
        st.caption("Latest Log Entry:")
        try:
            with open("prediction_logs.txt", "r") as f:
                last_line = f.readlines()[-1]
                st.code(last_line)
        except:
            st.write("No logs available yet.")

# -----------------------------
# 4. Main Interface
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.title(" Spam Detection System")
    st.subheader("Neural Text Analysis")
    
    user_message = st.text_area(
        "Analyze Communication:",
        placeholder="Paste the message content here...",
        height=200
    )
    
    predict_btn = st.button("Run Security Scan")

with col2:
    st.markdown("### System Metrics")
    st.write("Real-time classification confidence and latency.")
    placeholder = st.empty() # Used for updating result cards

# -----------------------------
# 5. Prediction Logic
# -----------------------------
if predict_btn:
    if not user_message.strip():
        st.warning(" Action Required: Please input text to analyze.")
    else:
        with st.status("Analyzing message patterns...", expanded=True) as status:
            start_time = time.time()
            
            # NLP Processing
            st.write("Vectorizing input...")
            message_vector = vectorizer.transform([user_message])
            
            st.write("Calculating probability...")
            prediction = model.predict(message_vector)[0]
            probability = model.predict_proba(message_vector)[0][1]
            
            latency = time.time() - start_time
            time.sleep(0.5) # Slight delay for UX
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # UI Results
        with col2:
            if prediction == 1:
                st.error("### SPAM DETECTED")
                st.metric("Risk Level", f"{probability:.1%}", delta="High Risk", delta_color="inverse")
            else:
                st.success("### VERIFIED HAM")
                st.metric("Trust Score", f"{1-probability:.1%}", delta="Safe")
            
            st.divider()
            st.caption(f"Latency: {latency:.4f}s | Engine: Scikit-learn")

        # Log prediction
        logging.basicConfig(filename="prediction_logs.txt", level=logging.INFO, format="%(asctime)s | %(message)s")
        logging.info(f"Msg: {user_message[:30]}... | Pred: {prediction} | Prob: {probability:.4f}")

st.markdown("---")
>>>>>>> ae3abbbc2e41e8ddf09d614a1558e0a879b3aacb
st.caption("© 2026 Sentinella Security Solutions | End-to-End Intelligent Spam Detection")