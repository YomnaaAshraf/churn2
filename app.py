# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification
# --- Page Configuration ---
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)
# Suppress a specific Streamlit warning about pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

# --- Resource Loading (Cached for Performance) ---
# This decorator loads the models and tools only once when the app starts.
@st.cache_resource
def load_assets():
    """Load all machine learning models and associated tools."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(base_path, "models")
    tools_path = os.path.join(base_path, "tools")

    assets = {}
    try:
        # Load traditional ML models
        assets["Logistic Regression"] = joblib.load(os.path.join(models_path, "logistic_regression_model.pkl"))
        assets["SVM"] = joblib.load(os.path.join(models_path, "svc_model.pkl"))
        assets["XGBoost"] = joblib.load(os.path.join(models_path, "best_xgb_model.pkl"))

        # --- THIS IS THE CORRECTED PART FOR THE LLM ---
        llm_model_path = os.path.join(models_path, "distilbert_model0")
        if not os.path.isdir(llm_model_path):
            st.error(f"DistilBERT model directory not found. Expected at: {llm_model_path}")
            return None
        
        # Explicitly load the model using its specific class, not the AutoModel class
        assets["DistilBERT LLM"] = DistilBertForSequenceClassification.from_pretrained(llm_model_path)
        assets["tokenizer"] = AutoTokenizer.from_pretrained(llm_model_path)
        # --- END OF CORRECTION ---

        # Load scalers and encoders
        assets["scaler_first"] = joblib.load(os.path.join(tools_path, "scaler_first.pkl"))
        assets["le_gender"] = joblib.load(os.path.join(tools_path, "le_gender.pkl"))
        assets["le_Contract"] = joblib.load(os.path.join(tools_path, "le_Contract.pkl"))
        assets["le_PaymentMethod"] = joblib.load(os.path.join(tools_path, "le_PaymentMethod.pkl"))
        assets["le_InternetService"] = joblib.load(os.path.join(tools_path, "le_InternetService.pkl"))

        return assets
    except FileNotFoundError as e:
        st.error(f"Fatal Error: A required model or tool file was not found. Please check your folder structure. Details: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading assets: {e}")
        return None

# --- Preprocessing & Prediction Functions ---

def preprocess_for_ml(df, assets):
    """Prepares raw dataframe for traditional ML model prediction."""
    df_processed = df.copy()
    
    # Handle TotalCharges - convert to numeric, coercing errors, and filling NaN with median
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    if df_processed['TotalCharges'].isnull().any():
        # Using a fixed median from training if available, otherwise calculate
        # For simplicity here, we calculate from the input if needed.
        median_charge = df_processed['TotalCharges'].median()
        df_processed['TotalCharges'].fillna(median_charge, inplace=True)

    # Binary columns
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

    # Service columns
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        df_processed[col] = df_processed[col].replace('No internet service', 'No').map({'Yes': 1, 'No': 0})
    
    df_processed['MultipleLines'] = df_processed['MultipleLines'].replace('No phone service', 'No').map({'Yes': 1, 'No': 0})
    
    # Categorical columns with LabelEncoders
    try:
        df_processed['gender'] = assets['le_gender'].transform(df_processed['gender'])
        df_processed['Contract'] = assets['le_Contract'].transform(df_processed['Contract'])
        df_processed['PaymentMethod'] = assets['le_PaymentMethod'].transform(df_processed['PaymentMethod'])
        df_processed['InternetService'] = assets['le_InternetService'].transform(df_processed['InternetService'])
    except Exception as e:
        st.error(f"Label Encoding Error: {e}. Check if input values are valid.")
        return None
        
    return df_processed

def convert_to_text(df_row):
    """Converts a single dataframe row to a descriptive text string for the LLM."""
    row = df_row.iloc[0]
    text = (f"User's gender is {row['gender']}, "
            f"{'is a senior citizen' if row['SeniorCitizen'] else 'is not a senior citizen'}, "
            f"{'has a partner' if row['Partner'] == 'Yes' else 'has no partner'}, "
            f"{'has dependents' if row['Dependents'] == 'Yes' else 'has no dependents'}, "
            f"has a tenure of {row['tenure']} months, "
            f"{'has phone service' if row['PhoneService'] == 'Yes' else 'has no phone service'}, "
            f"has {row['MultipleLines']} for multiple lines, "
            f"has {row['InternetService']} internet, "
            f"and has online security: {row['OnlineSecurity']}, "
            f"online backup: {row['OnlineBackup']}, "
            f"device protection: {row['DeviceProtection']}, "
            f"tech support: {row['TechSupport']}, "
            f"streaming TV: {row['StreamingTV']}, "
            f"and streaming movies: {row['StreamingMovies']}. "
            f"The contract is {row['Contract']}, with {row['PaperlessBilling']} paperless billing "
            f"and the {row['PaymentMethod']} payment method. "
            f"Monthly charges are ${float(row['MonthlyCharges']):.2f} and total charges are ${float(pd.to_numeric(row['TotalCharges'], errors='coerce')):.2f}.")
    return text

# --- Explainability Functions ---

def display_shap_explanation(model, preprocessed_data, model_name):
    """Generates and displays SHAP force plot for a single prediction."""
    st.subheader(f"Explainability for {model_name} (SHAP)")
    with st.spinner("Generating explanation..."):
        try:
            if model_name == "XGBoost":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(preprocessed_data)
            else: # Logistic Regression or SVM
                # Using LinearExplainer for coefficients. SVM is a linear kernel in your notebook.
                explainer = shap.LinearExplainer(model, preprocessed_data)
                shap_values = explainer.shap_values(preprocessed_data)

            # Force plot for a single instance
            fig, ax = plt.subplots()
            shap.force_plot(explainer.expected_value, shap_values[0,:], preprocessed_data.iloc[0,:], matplotlib=True, show=False)
            st.pyplot(fig, bbox_inches='tight', dpi=300, pad_inches=0)
            plt.close()
            
            st.info("The plot above shows which features pushed the prediction higher (in red) or lower (in blue).")
        except Exception as e:
            st.error(f"Could not generate SHAP plot. Error: {e}")

# --- Main App Interface ---
assets = load_assets()

if assets:
    st.sidebar.header("Input Method")
    input_method = st.sidebar.radio("Choose how to provide data:", ("Single Customer Form", "CSV File Upload"))

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Select a model for prediction:", list(assets.keys()))

    if input_method == "Single Customer Form":
        st.header("👤 Single Customer Prediction")
        
        with st.form(key='customer_form'):
            # Form fields
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                SeniorCitizen = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                Partner = st.radio("Partner", ["Yes", "No"])
                Dependents = st.radio("Dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (Months)", 0, 72, 12)
                PhoneService = st.radio("Phone Service", ["Yes", "No"])
                MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, format="%.2f")

            with col2:
                OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                PaperlessBilling = st.radio("Paperless Billing", ["Yes", "No"])
                PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, format="%.2f")
            
            submit_button = st.form_submit_button(label='Predict Churn')

        if submit_button:
            # Create a dataframe from the form input
            input_df = pd.DataFrame([{
                "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents,
                "tenure": tenure, "PhoneService": PhoneService, "MultipleLines": MultipleLines,
                "InternetService": InternetService, "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection, "TechSupport": TechSupport, "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies, "Contract": Contract, "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges
            }])
            
            prediction, churn_prob = 0, 0.0

            if model_choice == "DistilBERT LLM":
                with st.spinner("Converting data to text and predicting..."):
                    text_input = convert_to_text(input_df)
                    st.info(f"**Text generated for LLM:**\n> {text_input}")
                    tokenized = assets['tokenizer'](text_input, return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        logits = assets['DistilBERT LLM'](**tokenized).logits
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                    prediction = np.argmax(probs)
                    churn_prob = probs[1]
            else:
                with st.spinner("Preprocessing data and predicting..."):
                    processed_df = preprocess_for_ml(input_df, assets)
                    
                    # The scaler was likely fitted on unscaled data, so we scale here
                    processed_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = assets['scaler_first'].transform(processed_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

                    model = assets[model_choice]
                    prediction = model.predict(processed_df)[0]
                    churn_prob = model.predict_proba(processed_df)[0, 1]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"**Prediction: Churn** (Confidence: {churn_prob:.2%})")
            else:
                st.success(f"**Prediction: No Churn** (Confidence: {1 - churn_prob:.2%})")

            # Explainability
            if model_choice != "DistilBERT LLM":
                # Create a fresh preprocessed df for SHAP to avoid scaling issues
                processed_df_for_shap = preprocess_for_ml(input_df, assets)
                display_shap_explanation(assets[model_choice], processed_df_for_shap, model_choice, processed_df_for_shap.columns)
            else:
                st.info("SHAP is not configured for the LLM in this app. Explainability for LLMs often requires different techniques.")

    elif input_method == "CSV File Upload":
        st.header("📄 Batch Prediction via CSV Upload")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("First 5 rows of your uploaded data:")
            st.dataframe(df.head())

            if st.button("Run Batch Prediction"):
                # Similar logic as single prediction, but applied to the whole dataframe
                predictions = []
                churn_probs = []
                
                with st.spinner(f"Processing and predicting with {model_choice}..."):
                    if model_choice == "DistilBERT LLM":
                        text_inputs = [convert_to_text(pd.DataFrame([row])) for index, row in df.iterrows()]
                        tokenized = assets['tokenizer'](text_inputs, return_tensors="pt", padding=True, truncation=True)
                        with torch.no_grad():
                            logits = assets['DistilBERT LLM'](**tokenized).logits
                        probs = torch.softmax(logits, dim=1).numpy()
                        predictions = np.argmax(probs, axis=1)
                        churn_probs = probs[:, 1]
                    else:
                        processed_df = preprocess_for_ml(df, assets)
                        processed_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = assets['scaler_first'].transform(processed_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
                        model = assets[model_choice]
                        predictions = model.predict(processed_df)
                        churn_probs = model.predict_proba(processed_df)[:, 1]

                df_results = df.copy()
                df_results['Churn_Prediction'] = ["Yes" if p == 1 else "No" for p in predictions]
                df_results['Churn_Probability'] = [f"{p:.2%}" for p in churn_probs]
                
                st.subheader("Batch Prediction Results")
                st.dataframe(df_results)
                
                csv_output = df_results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", csv_output, "churn_predictions.csv", "text/csv")
