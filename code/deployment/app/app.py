import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Predictor")
st.markdown("""
Predict your chances of surviving the Titanic disaster based on passenger characteristics.
""")

with st.form("prediction_form"):
    st.header("Passenger Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sex = st.selectbox("Sex", ["female", "male"], help="Select your gender")
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
        fare = st.number_input("Fare (£)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    
    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    
    submitted = st.form_submit_button("Predict Survival", use_container_width=True)

if submitted:
    passenger_data = {
        "sex": sex,
        "age": age,
        "fare": fare,
        "sibsp": sibsp,
        "parch": parch
    }
    
    try:
        response = requests.post(
            "http://api:8000/predict",
            json=passenger_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            st.success("Prediction completed successfully!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Survival Prediction",
                    value=result["survival_status"],
                    delta=f"{result['confidence']:.1%} confidence"
                )
            
            with col2:
                survival_prob = result["survival_probability"]
                st.metric(
                    label="Survival Probability",
                    value=f"{survival_prob:.1%}"
                )
            
            st.progress(survival_prob)
            
            st.info(f"""
            **Prediction Details:**
            - Survival Chance: {survival_prob:.1%}
            - Confidence Level: {result['confidence']:.1%}
            - Final Prediction: {result['survival_status']}
            """)
            
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: Please make sure the API server is running. {e}")

st.sidebar.header("About")
st.sidebar.markdown("""
This application predicts survival chances on the Titanic using machine learning.

**Features used:**
- Sex
- Age  
- Fare paid
- Number of siblings/spouses
- Number of parents/children

The model was trained on historical Titanic passenger data.
""")

st.markdown("---")
st.markdown("*Kamilya Shakirova - PMLDL Assignment 1 - Titanic Survival Prediction*")