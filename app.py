# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import shap
import matplotlib.pyplot as plt
import os
import io

# --- This is the key change: We import the specific class directly ---
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# --- Page Configuration ---
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)
st.set_option('deprecation.showPyplotGlobalUse', False)

# --- Resource Loading (Cached for Performance) ---
@st.cache_resource
def load_assets():
    """Load all machine learning models and associated tools."""
    # Build relative paths from the script's location
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(base_path, "models")
    tools_path = os.path.join(base_path, "tools")

    assets = {}
    try:
        # Load traditional ML models
        assets["Logistic Regression"] = joblib.load(os.path.join(models_path, "logistic_regression_model.pkl"))
        assets["SVM"] = joblib.load(os.path.join(models_path, "svc_model.pkl"))
        assets["XGBoost"] = joblib.load(os.path.join(models_path, "best_xgb_model.pkl"))

        # Load LLM model and tokenizer from the saved directory
        llm_model_path = os.path.join(models_path, "distilbert_model0")
        if not os.path.isdir(llm_model_path):
            st.error(f"DistilBERT model directory not found. Expected at: {llm_model_path}")
            return None
        
        # BYPASSING AUTOMODEL: We are now explicitly telling transformers to use the DistilBert class.
        assets["DistilBERT LLM"] = DistilBertForSequenceClassification.from_pretrained(llm_model_path)
        assets["tokenizer"] = AutoTokenizer.from_pretrained(llm_model_path)

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
    
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    if df_processed['TotalCharges'].isnull().any():
        median_charge = df_processed['TotalCharges'].median() # Simple fallback
        df_processed['TotalCharges'].fillna(median_charge, inplace=True)

    # Map binary features
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

    # Consolidate and map service columns
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        df_processed[col] = df_processed[col].replace('No internet service', 'No').map({'Yes': 1, 'No': 0})
    
    df_processed['MultipleLines'] = df_processed['MultipleLines'].replace('No phone service', 'No').map({'Yes': 1, 'No': 0})
    
    # Apply label encoders
    try:
        df_processed['gender'] = assets['le_gender'].transform(df_processed['gender'])
        df_processed['Contract'] = assets['le_Contract'].transform(df_processed['Contract'])
        df_processed['PaymentMethod'] = assets['le_PaymentMethod'].transform(df_processed['PaymentMethod'])
        df_processed['InternetService'] = assets['le_InternetService'].transform(df_processed['InternetService'])
    except Exception as e:
        st.error(f"Label Encoding Error: {e}. Check if all categorical values in your input are present in the training data.")
        return None
        
    # Scale numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_processed[numerical_cols] = assets['scaler_first'].transform(df_processed[numerical_cols])

    # Ensure final columns match the training order for traditional models
    final_columns = assets['XGBoost'].get_booster().feature_names
    df_processed = df_processed[final_columns]
    
    return df_processed

def convert_to_text(df_row):
    """Converts a single dataframe row to a descriptive text string for the LLM."""
    row = df_row.iloc[0]
    total_charges_val = pd.to_numeric(row['TotalCharges'], errors='coerce')
    total_charges_str = f"{total_charges_val:.2f}" if pd.notnull(total_charges_val) else "0.00"

    text = (f"User's gender is {row['gender']}, "
            f"{'is a senior citizen' if row['SeniorCitizen'] else 'is not a senior citizen'}, "
            f"{'has a partner' if row['Partner'] == 'Yes' else 'has no partner'}, "
            f"{'has dependents' if row['Dependents'] == 'Yes' else 'has no dependents'}, "
            f"has a tenure of {row['tenure']} months, "
            f"{'has phone service' if row['PhoneService'] == 'Yes' else 'has no phone service'}, "
            f"has {row['MultipleLines']} for multiple lines, "
            f"has {row['InternetService']} internet service, "
            f"and has online security: {row['OnlineSecurity']}, "
            f"online backup: {row['OnlineBackup']}, "
            f"device protection: {row['DeviceProtection']}, "
            f"tech support: {row['TechSupport']}, "
            f"streaming TV: {row['StreamingTV']}, "
            f"and streaming movies: {row['StreamingMovies']}. "
            f"The contract is {row['Contract']}, with {row['PaperlessBilling']} paperless billing "
            f"and the {row['PaymentMethod']} payment method. "
            f"Monthly charges are ${float(row['MonthlyCharges']):.2f} and total charges are ${total_charges_str}.")
    return text

def predict_llm(texts, resources):
    """Generates predictions and probabilities from the LLM."""
    tokenizer = resources['tokenizer']
    model = assets['DistilBERT LLM']
    
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = torch.softmax(logits, dim=1)
    return probs.numpy()


# --- Explainability Functions ---
def display_shap_explanation(model, processed_data, model_name):
    """Generates and displays SHAP force plot for a single prediction."""
    st.subheader(f"Explainability for {model_name} (SHAP)")
    with st.spinner("Generating explanation..."):
        try:
            if model_name == "XGBoost":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(processed_data)
                expected_value = explainer.expected_value
            else: 
                explainer = shap.LinearExplainer(model, processed_data)
                shap_values = explainer.shap_values(processed_data)
                expected_value = explainer.expected_value[0] # LinearExplainer returns a list

            st.write("This plot shows which features pushed the prediction higher (in red) or lower (in blue) than the base value.")
            
            # Use st.pyplot for the force plot
            fig, ax = plt.subplots()
            shap.force_plot(
                expected_value,
                shap_values[0,:],
                processed_data.iloc[0,:],
                matplotlib=True,
                show=False,
                figsize=(15, 5) # Adjust size for better readability
            )
            st.pyplot(fig, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            st.error(f"Could not generate SHAP plot for this model. Error: {e}")

# --- Main App Interface ---
assets = load_assets()

if assets:
    st.sidebar.title("Controls")
    model_choice = st.sidebar.selectbox("Choose a Model", list(assets.keys()))
    input_method = st.sidebar.radio("Choose Input Method", ["Manual Input (Single Customer)", "CSV File Upload"])

    if input_method == "Manual Input (Single Customer)":
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
                
            with col2:
                OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

            st.markdown("---")
            col3, col4 = st.columns(2)
            with col3:
                Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                PaperlessBilling = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
                PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            with col4:
                MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, format="%.2f")
                TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, format="%.2f")

            submit_button = st.form_submit_button(label='Predict Churn')

        if submit_button:
            input_df = pd.DataFrame([locals()])
            
            prediction, churn_prob = 0, 0.0

            st.subheader("Prediction Result")
            with st.spinner(f"Predicting using {model_choice}..."):
                if model_choice == "DistilBERT LLM":
                    text_input = convert_to_text(input_df)
                    st.info(f"**Text generated for LLM:**\n> {text_input}")
                    probs = predict_llm([text_input], assets)[0]
                    prediction = np.argmax(probs)
                    churn_prob = probs[1]
                else: 
                    processed_df = preprocess_for_ml(input_df, assets)
                    model = assets[model_choice]
                    prediction = model.predict(processed_df)[0]
                    churn_prob = model.predict_proba(processed_df)[0, 1]

            if prediction == 1:
                st.error(f"**Prediction: Churn** (Confidence: {churn_prob:.2%})")
            else:
                st.success(f"**Prediction: No Churn** (Confidence: {1-churn_prob:.2%})")

            if model_choice != "DistilBERT LLM":
                display_shap_explanation(assets[model_choice], preprocess_for_ml(input_df.copy(), assets), model_choice)
            else:
                st.info("SHAP explainability for the LLM is not implemented in this app.")

    elif input_method == "CSV File Upload":
        st.header("📄 Batch Prediction via CSV Upload")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.write("First 5 rows of your uploaded data:")
            st.dataframe(df_batch.head())

            if st.button(f"Run Batch Prediction with {model_choice}"):
                with st.spinner(f"Processing and predicting..."):
                    if model_choice == "DistilBERT LLM":
                        text_inputs = [convert_to_text(pd.DataFrame([row])) for _, row in df_batch.iterrows()]
                        probs_batch = predict_llm(text_inputs, assets)
                        predictions = np.argmax(probs_batch, axis=1)
                        churn_probs = probs_batch[:, 1]
                    else:
                        processed_df = preprocess_for_ml(df_batch, assets)
                        model = assets[model_choice]
                        predictions = model.predict(processed_df)
                        churn_probs = model.predict_proba(processed_df)[:, 1]

                results_df = df_batch.copy()
                results_df['Churn_Prediction'] = ["Yes" if p == 1 else "No" for p in predictions]
                results_df['Churn_Probability'] = [f"{p:.2%}" for p in churn_probs]
                
                st.subheader("Batch Prediction Results")
                st.dataframe(results_df)

                # Convert dataframe to CSV for download
                output = io.StringIO()
                results_df.to_csv(output, index=False)
                csv_output = output.getvalue()
                
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_output,
                    file_name=f'churn_predictions_{model_choice.replace(" ", "_")}.csv',
                    mime='text/csv',
                )
