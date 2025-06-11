import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import shap
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import shap
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
# --- Page Configuration ---
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide"
)
st.set_option('deprecation.showPyplotGlobalUse', False)

script_dir = os.path.dirname(os.path.abspath("C:\Users\ducci\Downloads\notebook-20250611T175856Z-1-001\notebook\app.py"))

@st.cache_resource
def load_models_and_tools():
    """Load all models, tokenizers, scalers, and encoders once."""
    
    # --- Step 1: Build the full, correct path for every file ---
    # This combines your script's location with the relative folder/file names
    lr_path = os.path.join(script_dir, "models", "logistic_regression_model.pkl")
    svc_path = os.path.join(script_dir, "models", "svc_model.pkl")
    xgb_path = os.path.join(script_dir, "models", "best_xgb_model.pkl")
    llm_path = os.path.join(script_dir, "models", "distilbert_model0") # Path to the FOLDER
    scaler_path = os.path.join(script_dir, "tools", "scaler_first.pkl")
    le_gender_path = os.path.join(script_dir, "tools", "le_gender.pkl")
    le_Contract_path = os.path.join(script_dir, "tools", "le_Contract.pkl")
    le_PaymentMethod_path = os.path.join(script_dir, "tools", "le_PaymentMethod.pkl")
    le_InternetService_path = os.path.join(script_dir, "tools", "le_InternetService.pkl")
    
    # --- Step 2: Now, load all assets using the path variables you just created ---
    
    # Traditional ML Models
    lr_model = joblib.load(lr_path)
    svc_model = joblib.load(svc_path)
    xgb_model = joblib.load(xgb_path)

    # LLM Model
    llm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, num_labels=2)
    
    # Scalers
    scaler_first = joblib.load(scaler_path)
    
    # Label Encoders
    le_gender = joblib.load(le_gender_path)
    le_Contract = joblib.load(le_Contract_path)
    le_PaymentMethod = joblib.load(le_PaymentMethod_path)
    le_InternetService = joblib.load(le_InternetService_path)
    
    return {
        "Logistic Regression": lr_model,
        "SVM": svc_model,
        "XGBoost": xgb_model,
        "DistilBERT LLM": llm_model,
        "tokenizer": llm_tokenizer,
        "scaler": scaler_first,
        "le_gender": le_gender,
        "le_Contract": le_Contract,
        "le_PaymentMethod": le_PaymentMethod,
        "le_InternetService": le_InternetService
    }

# Load all assets
assets = load_models_and_tools()

# --- Preprocessing Functions ---
def preprocess_for_traditional_models(df, assets):
    """Prepares raw data for Logistic Regression, SVM, and XGBoost models."""
    df_processed = df.copy()
    
    # Clean TotalCharges
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    df_processed['TotalCharges'] = df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median())

    # Map binary features
    binary_map = {'Yes': 1, 'No': 0}
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df_processed[col] = df_processed[col].map(binary_map)

    # Clean and map service columns
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        df_processed[col] = df_processed[col].replace('No internet service', 'No').map(binary_map)
    df_processed['MultipleLines'] = df_processed['MultipleLines'].replace('No phone service', 'No').map(binary_map)
    
    # Apply label encoders
    df_processed['gender'] = assets['le_gender'].transform(df_processed['gender'])
    df_processed['Contract'] = assets['le_Contract'].transform(df_processed['Contract'])
    df_processed['PaymentMethod'] = assets['le_PaymentMethod'].transform(df_processed['PaymentMethod'])
    df_processed['InternetService'] = assets['le_InternetService'].transform(df_processed['InternetService'])
    
    # Ensure correct order of columns for the model
    model_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                     'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                     'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                     'MonthlyCharges', 'TotalCharges']
    return df_processed[model_columns]

def convert_to_text(df):
    """Converts a DataFrame row into a text string for the LLM."""
    row = df.iloc[0]
    st = f"User's gender is {row['gender']}, "
    st += "user is senior citizen, " if row['SeniorCitizen'] else "user is not senior citizen, "
    st += f"user has partner, " if row['Partner'] == 'Yes' else "user has no partner, "
    st += f"user has dependents, " if row['Dependents'] == 'Yes' else "user has no dependents, "
    st += f"user's tenure is {row['tenure']}, "
    st += "user has phone service, " if row['PhoneService'] == 'Yes' else "user has no phone service, "
    st += f"user has {row['MultipleLines']} multiple lines, ".lower()
    st += f"user has {row['InternetService']} internet service, ".lower()
    st += "user has online security, " if row['OnlineSecurity'] == 'Yes' else "user has no online security, "
    st += "user has online backup, " if row['OnlineBackup'] == 'Yes' else "user has no online backup, "
    st += "user has device protection, " if row['DeviceProtection'] == 'Yes' else "user has no device protection, "
    st += "user has tech support, " if row['TechSupport'] == 'Yes' else "user has no tech support, "
    st += "user has streaming tv, " if row['StreamingTV'] == 'Yes' else "user has no streaming tv, "
    st += "user has streaming movies, " if row['StreamingMovies'] == 'Yes' else "user has no streaming movies, "
    st += f"user's contract is {row['Contract']}, "
    st += "user has paperless billing, " if row['PaperlessBilling'] == 'Yes' else "user has no paperless billing, "
    st += f"user's payment method is {row['PaymentMethod']}, "
    st += f"user's monthly charge is {row['MonthlyCharges']}, "
    st += f"user's total charges is {row['TotalCharges']}."
    return st.replace('No internet service', 'no internet service').replace('No phone service', 'no phone service')


# --- SHAP and Explainability Functions ---
def get_shap_plot(model, model_name, preprocessed_input, column_names):
    st.subheader("Model Explainability (SHAP)")
    with st.spinner("Generating SHAP explanation..."):
        if model_name == "XGBoost":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(preprocessed_input)
        elif model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(model, preprocessed_input)
            shap_values = explainer.shap_values(preprocessed_input)
        elif model_name == "SVM":
            st.warning("SHAP for SVM is computationally intensive. Using a simplified background for approximation.")
            # For performance, use a summarized or sampled background. Here we just use the input itself.
            explainer = shap.KernelExplainer(model.predict_proba, preprocessed_input)
            shap_values = explainer.shap_values(preprocessed_input, nsamples=100)[1] # for class 1
        
        st.write("This plot shows which features pushed the prediction towards or away from churning.")
        st.write("Features in **red** increase the probability of churn. Features in **blue** decrease it.")
        
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value if model_name != "SVM" else explainer.expected_value[1],
                        shap_values,
                        preprocessed_input,
                        feature_names=column_names,
                        matplotlib=True,
                        show=False)
        st.pyplot(fig, bbox_inches='tight')
        plt.close(fig)


# --- UI Layout ---
st.title("📡 Telco Customer Churn Prediction")
st.markdown("This app simulates a deployment environment for predicting customer churn using four different models. Choose an input method and a model to get a prediction and an explanation.")

# Sidebar for controls
st.sidebar.title("Controls")
model_choice = st.sidebar.selectbox("Choose a Model", ["XGBoost", "Logistic Regression", "SVM", "DistilBERT LLM"])
input_method = st.sidebar.radio("Choose Input Method", ["Manual Input (Single Customer)", "Upload CSV (Batch Prediction)"])

# --- Main App Logic ---
if input_method == "Manual Input (Single Customer)":
    st.header("Enter Customer Details")
    
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            Partner = st.selectbox("Has Partner?", ["Yes", "No"])
            Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            PhoneService = st.selectbox("Phone Service?", ["Yes", "No"])
            
        with col2:
            MultipleLines = st.selectbox("Multiple Lines?", ["Yes", "No", "No phone service"])
            InternetService = st.selectbox("Internet Service?", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security?", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup?", ["Yes", "No", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection?", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Tech Support?", ["Yes", "No", "No internet service"])

        with col3:
            StreamingTV = st.selectbox("Streaming TV?", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies?", ["Yes", "No", "No internet service"])
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
            TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1500.0)

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        input_data = pd.DataFrame([locals()], columns=list(locals().keys()))
        
        st.subheader("Prediction Result")
        
        if model_choice == "DistilBERT LLM":
            text_input = convert_to_text(input_data)
            st.info(f"**Text sent to LLM:** {text_input}")
            
            with st.spinner(f"Asking {model_choice} for a prediction..."):
                tokenizer = assets['tokenizer']
                model = assets['DistilBERT LLM']
                tokenized_input = tokenizer(text_input, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**tokenized_input).logits
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                
                prediction = np.argmax(probs)
                churn_prob = probs[1]

        else: # Traditional ML models
            preprocessed_data = preprocess_for_traditional_models(input_data, assets)
            scaled_data = assets['scaler'].transform(preprocessed_data)
            model = assets[model_choice]

            with st.spinner(f"Using {model_choice} for prediction..."):
                prediction = model.predict(scaled_data)[0]
                probs = model.predict_proba(scaled_data)[0]
                churn_prob = probs[1]
        
        # Display Result
        if prediction == 1:
            st.error(f"Prediction: **Churn** (Probability: {churn_prob:.2%})")
        else:
            st.success(f"Prediction: **No Churn** (Probability: {1-churn_prob:.2%})")
            
        # Display SHAP plot for traditional models
        if model_choice != "DistilBERT LLM":
            get_shap_plot(model, model_choice, pd.DataFrame(scaled_data, columns=preprocessed_data.columns), preprocessed_data.columns)

else: # CSV Upload
    st.header("Upload Customer Data (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Sample:")
        st.dataframe(df.head())

        if st.button("Predict on this file"):
            if model_choice == "DistilBERT LLM":
                 st.warning("Batch prediction for LLM is slow and processed row-by-row. This might take a while.")
                 predictions = []
                 churn_probs = []
                 with st.spinner(f"Predicting with {model_choice}..."):
                     for i, row in df.iterrows():
                         row_df = pd.DataFrame([row])
                         text_input = convert_to_text(row_df)
                         tokenizer = assets['tokenizer']
                         model = assets['DistilBERT LLM']
                         tokenized_input = tokenizer(text_input, return_tensors="pt")
                         with torch.no_grad():
                             logits = model(**tokenized_input).logits
                             probs = torch.softmax(logits, dim=1).numpy()[0]
                         predictions.append(np.argmax(probs))
                         churn_probs.append(probs[1])

            else: # Traditional models
                with st.spinner(f"Preprocessing and predicting with {model_choice}..."):
                    preprocessed_df = preprocess_for_traditional_models(df, assets)
                    scaled_df = assets['scaler'].transform(preprocessed_df)
                    model = assets[model_choice]
                    predictions = model.predict(scaled_df)
                    churn_probs = model.predict_proba(scaled_df)[:, 1]

            # Display results
            results_df = df.copy()
            results_df['Predicted Churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
            results_df['Churn Probability'] = [f"{p:.2%}" for p in churn_probs]
            
            st.subheader("Prediction Results")
            st.dataframe(results_df)

            st.download_button(
                label="Download Results as CSV",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name=f'churn_predictions_{model_choice}.csv',
                mime='text/csv',
            )
            
            # Explain first row for traditional models
            if model_choice != "DistilBERT LLM":
                st.markdown("---")
                st.subheader("Explanation for the First Row")
                get_shap_plot(model, model_choice, pd.DataFrame(scaled_df[0:1], columns=preprocessed_df.columns), preprocessed_df.columns)
