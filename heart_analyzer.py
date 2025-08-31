# heart_app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# === Load Dataset and Train Model ===
data = pd.read_csv(r"C:\Users\rjana\OneDrive\Downloads\Documents\Desktop\program\AI\capston-ai\heart.csv")
X = data.drop("target", axis=1)
y = data["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = SVC(kernel='rbf', C=10, gamma=0.001, probability=True)
model.fit(X_train, y_train)

# === Page Config ===
st.set_page_config(page_title="💖 Heart Risk Analyzer", layout="wide")
st.title("💓 Heart Disease Risk Prediction System")

st.markdown("This tool helps you assess your **heart disease risk** using common medical parameters. "
            "Don’t worry if you’re not familiar with the terms — helpful explanations are shown beside each input. 💡")

# === Sidebar Info ===
st.sidebar.header("📘 Medical Abbreviations Explained")

with st.sidebar.expander("ℹ️ Age (age)"):
    st.markdown("""
    Entered directly in **years**  
    → help: Risk of heart disease increases with age
    """)

with st.sidebar.expander("ℹ️ Sex (sex)"):
    st.markdown("""
    **1 = Male** 🧑 → help: Biological male  
    **0 = Female** 👩 → help: Biological female  
    """)

with st.sidebar.expander("ℹ️ Chest Pain Type (cp)"):
    st.markdown("""
    **0 = No Chest Pain** → help: No pain felt  
    **1 = Typical Angina** → help: Classic chest pain during activity  
    **2 = Atypical Angina** → help: Unusual chest discomfort  
    **3 = Non-anginal Pain** → help: Not related to heart (e.g. muscle pain)  
    **4 = Asymptomatic** → help: No chest pain, but disease may still exist  
    """)

with st.sidebar.expander("ℹ️ Resting Blood Pressure (trestbps)"):
    st.markdown("""
    Measured in **mmHg**  
    → help: Normal value is around 120/80 mmHg
    """)

with st.sidebar.expander("ℹ️ Cholesterol (chol)"):
    st.markdown("""
    Measured in **mg/dl**  
    → help: High values (>200) increase heart risk
    """)

with st.sidebar.expander("ℹ️ Fasting Blood Sugar (fbs)"):
    st.markdown("""
    **1 = Yes** → help: Fasting sugar > 120 mg/dl (possible diabetes risk)  
    **0 = No** → help: Normal sugar levels  
    """)

with st.sidebar.expander("ℹ️ Resting ECG (restecg)"):
    st.markdown("""
    **0 = Normal** → help: Normal ECG results  
    **1 = Abnormality** → help: Possible irregularities in rhythm  
    **2 = LVH** → help: Left Ventricular Hypertrophy (possible enlarged heart)  
    """)

with st.sidebar.expander("ℹ️ Maximum Heart Rate (thalach)"):
    st.markdown("""
    Highest heart rate during exercise  
    → help: Higher = healthier heart
    """)

with st.sidebar.expander("ℹ️ Exercise Induced Angina (exang)"):
    st.markdown("""
    **1 = Yes** → help: Chest pain occurs during exercise  
    **0 = No** → help: No chest pain during exercise  
    """)

with st.sidebar.expander("ℹ️ ST Depression (oldpeak)"):
    st.markdown("""
    Value of ST depression relative to rest  
    → help: Higher = more strain on heart
    """)

with st.sidebar.expander("ℹ️ Slope of ST Segment (slope)"):
    st.markdown("""
    **0 = Upsloping** → help: Usually healthy pattern  
    **1 = Flat** → help: Moderate concern  
    **2 = Downsloping** → help: Higher risk  
    """)

with st.sidebar.expander("ℹ️ Major Vessels Colored (ca)"):
    st.markdown("""
    **0–3** → help: Number of major vessels blocked. More blocked = higher risk  
    """)

with st.sidebar.expander("ℹ️ Thalassemia (thal)"):
    st.markdown("""
    **3 = Normal** → help: Normal blood flow  
    **6 = Fixed Defect** → help: Permanent blood flow issue  
    **7 = Reversible Defect** → help: Improves with treatment  
    """)

# === Input Section ===
st.header("🧾 Enter Your Health Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, 40)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type",
                      ["No Chest Pain", "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120mg/dl?", ["Yes", "No"])

with col2:
    restecg = st.selectbox("Resting ECG", ["Normal", "Abnormality", "LVH"])
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina?", ["Yes", "No"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# === Convert to numerical format ===
sex_val = 1 if sex == "Male" else 0
cp_val = ["No Chest Pain", "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
fbs_val = 1 if fbs == "Yes" else 0
restecg_val = ["Normal", "Abnormality", "LVH"].index(restecg)
exang_val = 1 if exang == "Yes" else 0
slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal_val = [3, 6, 7][["Normal", "Fixed Defect", "Reversible Defect"].index(thal)]

input_values = [age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                thalach, exang_val, oldpeak, slope_val, ca, thal_val]

# === Feature cause descriptions ===
cause_description = {
    "cp": "Chest pain is strongly linked to heart problems. More severe pain types indicate higher concern.",
    "chol": "High cholesterol can clog arteries, increasing heart disease risk.",
    "thalach": "A lower maximum heart rate during exercise may show limited heart performance.",
    "oldpeak": "Higher values mean greater strain on the heart during exercise.",
    "ca": "More blocked vessels indicate higher blockage and poorer blood flow.",
    "thal": "Certain thalassemia types can affect oxygen delivery and heart efficiency."
}

# === Prediction Button ===
if st.button("🔍 Predict My Heart Risk"):
    input_scaled = scaler.transform([input_values])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Result message
    result = "❤️ Heart Disease Detected" if prediction == 1 else "💚 No Heart Disease"
    st.subheader(result)

    # Health score as metric
    health_score = round((1 - probability) * 100)
    st.metric(label="🧬 Health Score", value=f"{health_score}%")

    # Likely contributing factor
    feature_importance = {
        "cp": input_values[2],
        "chol": input_values[4],
        "thalach": 220 - input_values[7],  # reversed (lower thalach = riskier)
        "oldpeak": input_values[9],
        "ca": input_values[11],
        "thal": input_values[12]
    }
    main_cause = max(feature_importance, key=feature_importance.get)
    st.info(f"🔎 Likely contributing factor: {cause_description[main_cause]}")

    # Guidance messages
    if prediction == 1:
        st.warning("⚠️ You may have a possibility of heart risk. Please consult a doctor for professional advice.")
    else:
        st.success("✅ Your result shows no detected heart disease. Still, regular checkups are important for good health.")

    st.caption("ℹ️ This is an AI-based tool for awareness and early screening. It is not a substitute for a medical diagnosis.")
