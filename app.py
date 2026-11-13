import joblib
import pandas as pd
import streamlit as st

CREATED_BY = "Syeda Zaibunnisa"

st.set_page_config(page_title="Employee Churn Predictor", page_icon="👥")


@st.cache_resource
def load_artifacts():
    schema = joblib.load("schema.pkl")
    pipeline = joblib.load("pipeline.pkl")
    return schema, pipeline


schema, pipeline = load_artifacts()

st.title("Employee Churn Predictor")
st.subheader(f"Created by {CREATED_BY}")
st.write("Fill in the employee details and press **Predict** to see whether the employee is likely to leave.")

inputs = {}
with st.form("employee_form"):
    # Numerical features: dropdown if few distinct values, otherwise number input/slider
    for col, meta in schema["numerical"].items():
        label = col
        if meta.get("choices"):
            choices = [int(c) if meta["is_integer"] else c for c in meta["choices"]]
            inputs[col] = st.selectbox(label, choices)
        elif meta["is_integer"]:
            lo, hi = int(meta["min"]), int(meta["max"])
            inputs[col] = st.slider(label, lo, hi, int(round(meta["median"])))
        else:
            inputs[col] = st.number_input(
                label, float(meta["min"]), float(meta["max"]), float(meta["median"])
            )

    # Categorical features: dropdown of values seen in training data
    for col, meta in schema["categorical"].items():
        inputs[col] = st.selectbox(col, meta["options"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # keep the same column order as training
    columns = list(schema["numerical"].keys()) + list(schema["categorical"].keys())
    row = pd.DataFrame([inputs])[columns]

    pred = int(pipeline.predict(row)[0])
    proba = float(pipeline.predict_proba(row)[0][1])

    st.markdown("### Prediction")
    if pred == 1:
        st.error(f"Likely to LEAVE (probability of leaving: {proba:.1%})")
    else:
        st.success(f"Likely to STAY (probability of leaving: {proba:.1%})")
    st.progress(min(max(proba, 0.0), 1.0))

    with st.expander("Input values used"):
        st.dataframe(row)
