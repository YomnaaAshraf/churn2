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

# Define script directory for file paths
script_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models_and_tools():
    """Load all models, tokenizers, scalers, and encoders once."""
    try:
        # Define file paths
        lr_path = os.path.join(script_dir, "models", "logistic_regression_model.pkl")
        svc_path = os.path.join(script_dir, "models", "svc_model.pkl")
        xgb_path = os.path.join(script_dir, "models", "best_xgb_model.pkl")
        llm_path = os.path.join(script_dir, "models", "distilbert_model0")
        scaler_path = os.path.join(script_dir, "tools", "scaler_first.pkl")
        le_gender_path = os.path.join(script_dir, "tools", "le_gender.pkl")
        le_Contract_path = os.path.join(script_dir, "tools", "le_Contract.pkl")
        le_PaymentMethod_path = os.path.join(script_dir, "tools", "le_PaymentMethod.pkl")
        le_InternetService_path = os.path.join(script_dir, "tools", "le_InternetService.pkl")

        # Load models and tools with error checking
        for path, name in [
            (lr_path, "Logistic Regression model"),
            (svc_path, "SVM model"),
            (xgb_path, "XGBoost model"),
            (scaler_path, "Scaler"),
            (le_gender_path, "Gender Label Encoder"),
            (le_Contract_path, "Contract Label Encoder"),
            (le_PaymentMethod_path, "Payment Method Label Encoder"),
            (le_InternetService_path, "Internet Service Label Encoder")
        ]:
            if not os.path.exists(path):
                st.error(f"{name} file not found at {path}")
                raise FileNotFoundError(f"File not found: {path}")

        lr_model = joblib.load(lr_path)
        svc_model = joblib.load(svc_path)
        xgb_model = joblib.load(xgb_path)

        # Load DistilBERT
        if not os.path.exists(llm_path):
            st.error(f"DistilBERT model directory not found at {llm_path}")
            raise FileNotFoundError(f"Directory not found: {llm_path}")
        llm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, num_labels=2)

        # Load scalers and encoders
        scaler_first = joblib.load(scaler_path)
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
    except Exception as e:
        st.error(f"Error loading models or tools: {str(e)}")
        return None

# Load all assets
assets = load_models_and_tools()
if assets is None:
    st.stop()

# --- Preprocessing Functions ---
@st.cache_data
def preprocess_for_traditional_models(df, assets):
    """Prepares raw data for Logistic Regression, SVM, and XGBoost models."""
    required_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return None

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
    try:
        df_processed['gender'] = assets['le_gender'].transform(df_processed['gender'])
        df_processed['Contract'] = assets['le_Contract'].transform(df_processed['Contract'])
        df_processed['PaymentMethod'] = assets['le_PaymentMethod'].transform(df_processed['PaymentMethod'])
        df_processed['InternetService'] = assets['le_InternetService'].transform(df_processed['InternetService'])
    except ValueError as e:
        st.error(f"Error in label encoding: {str(e)}. Ensure input values match training data categories.")
        return None

    return df_processed[required_columns]

@st.cache_data
def convert_to_text(df):
    """Converts a DataFrame row into a text string for the LLM."""
    if len(df) != 1:
        st.error("LLM input requires exactly one row of data.")
        return None
    required_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns for LLM: {', '.join(missing_cols)}")
        return None

    row = df.iloc[0]
    st_text = f"User's gender is {row['gender']}, "
    st_text += "user is senior citizen, " if row['SeniorCitizen'] else "user is not senior citizen, "
    st_text += f"user has partner, " if row['Partner'] == 'Yes' else "user has no partner, "
    st_text += f"user has dependents, " if row['Dependents'] == 'Yes' else "user has no dependents, "
    st_text += f"user's tenure is {row['tenure']}, "
    st_text += "user has phone service, " if row['PhoneService'] == 'Yes' else "user has no phone service, "
    st_text += f"user has {row['MultipleLines']} multiple lines, ".lower()
    st_text += f"user has {row['InternetService']} internet service, ".lower()
    st_text += "user has online security, " if row['OnlineSecurity'] == 'Yes' else "user has no online security, "
    st_text += "user has online backup, " if row['OnlineBackup'] == 'Yes' else "user has no online backup, "
    st_text += "user has device protection, " if row['DeviceProtection'] == 'Yes' else "user has no device protection, "
    st_text += "user has tech support, " if row['TechSupport'] == 'Yes' else "user has no tech support, "
    st_text += "user has streaming tv, " if row['StreamingTV'] == 'Yes' else "user has no streaming tv, "
    st_text += "user has streaming movies, " if row['StreamingMovies'] == 'Yes' else "user has no streaming movies, "
    st_text += f"user's contract is {row['Contract']}, "
    st_text += "user has paperless billing, " if row['PaperlessBilling'] == 'Yes' else "user has no paperless billing, "
    st_text += f"user's payment method is {row['PaymentMethod']}, "
    st_text += f"user's monthly charge is {row['MonthlyCharges']}, "
    st_text += f"user's total charges is {row['TotalCharges']}."
    return st_text.replace('No internet service', 'no internet service').replace('No phone service', 'no phone service')

# --- SHAP and Explainability Functions ---
def get_shap_plot(model, model_name, preprocessed_input, column_names):
    """Generate SHAP force plot for traditional ML models."""
    st.subheader("Model Explainability (SHAP)")
    with st.spinner("Generating SHAP explanation..."):
        try:
            if model_name == "XGBoost":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(preprocessed_input)
                expected_value = explainer.expected_value
            elif model_name == "Logistic Regression":
                explainer = shap.LinearExplainer(model, preprocessed_input)
                shap_values = explainer.shap_values(preprocessed_input)
                expected_value = explainer.expected_value
            elif model_name == "SVM":
                st.warning("SHAP for SVM is computationally intensive. Using input as background for approximation.")
                explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], preprocessed_input)
                shap_values = explainer.shap_values(preprocessed_input, nsamples=100)
                expected_value = explainer.expected_value

            st.write("This plot shows which features pushed the prediction towards or away from churning.")
            st.write("Features in **red** increase the probability of churn. Features in **blue** decrease it.")

            fig, ax = plt.subplots()
            shap.force_plot(
                expected_value,
                shap_values,
                preprocessed_input,
                feature_names=column_names,
                matplotlib=True,
                show=False
            )
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error generating SHAP plot: {str(e)}")

# --- UI Layout ---
st.title("📡 Telco Customer Churn Prediction")
st.markdown("This app predicts customer churn using four different models. Choose an input method and a model to get a prediction and an explanation.")

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
            gender = st.selectbox("Gender", ["Male", "Female"], help="Select the customer's gender.")
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Is the customer a senior citizen?")
            Partner = st.selectbox("Has Partner?", ["Yes", "No"], help="Does the customer have a partner?")
            Dependents = st.selectbox("Has Dependents?", ["Yes", "No"], help="Does the customer have dependents?")
            tenure = st.slider("Tenure (months)", 0, 72, 12, help="Number of months the customer has been with the company.")
            PhoneService = st.selectbox("Phone Service?", ["Yes", "No"], help="Does the customer have phone service?")

        with col2:
            MultipleLines = st.selectbox("Multiple Lines?", ["Yes", "No", "No phone service"], help="Does the customer have multiple phone lines?")
            InternetService = st.selectbox("Internet Service?", ["DSL", "Fiber optic", "No"], help="Type of internet service.")
            OnlineSecurity = st.selectbox("Online Security?", ["Yes", "No", "No internet service"], help="Does the customer have online security?")
            OnlineBackup = st.selectbox("Online Backup?", ["Yes", "No", "No internet service"], help="Does the customer have online backup?")
            DeviceProtection = st.selectbox("Device Protection?", ["Yes", "No", "No internet service"], help="Does the customer have device protection?")
            TechSupport = st.selectbox("Tech Support?", ["Yes", "No", "No internet service"], help="Does the customer have tech support?")

        with col3:
            StreamingTV = st.selectbox("Streaming TV?", ["Yes", "No", "No internet service"], help="Does the customer have streaming TV?")
            StreamingMovies = st.selectbox("Streaming Movies?", ["Yes", "No", "No internet service"], help="Does the customer have streaming movies?")
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], help="Type of contract.")
            PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"], help="Does the customer use paperless billing?")
            PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], help="Payment method.")
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, help="Monthly charges in dollars.")
            TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1500.0, help="Total charges in dollars.")

        col_submit, col_reset = st.columns(2)
        with col_submit:
            submitted = st.form_submit_button("Predict Churn")
        with col_reset:
            reset = st.form_submit_button("Reset Form")

    if reset:
        st.experimental_rerun()

    if submitted:
        # Input validation
        if MonthlyCharges < 0 or TotalCharges < 0:
            st.error("Monthly and Total Charges must be non-negative.")
        elif tenure < 0:
            st.error("Tenure must be non-negative.")
        else:
            input_data = pd.DataFrame([{
                'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Partner': Partner, 'Dependents': Dependents,
                'tenure': tenure, 'PhoneService': PhoneService, 'MultipleLines': MultipleLines,
                'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
                'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport, 'StreamingTV': StreamingTV,
                'StreamingMovies': StreamingMovies, 'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
                'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges
            }])

            st.subheader("Prediction Result")

            if model_choice == "DistilBERT LLM":
                text_input = convert_to_text(input_data)
                if text_input is None:
                    st.stop()
                st.info(f"**Text sent to LLM:** {text_input}")

                with st.spinner(f"Asking {model_choice} for a prediction..."):
                    tokenizer = assets['tokenizer']
                    model = assets['DistilBERT LLM']
                    tokenized_input = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        logits = model(**tokenized_input).logits
                        probs = torch.softmax(logits, dim=1).numpy()[0]

                    prediction = np.argmax(probs)
                    churn_prob = probs[1]

            else:  # Traditional ML models
                preprocessed_data = preprocess_for_traditional_models(input_data, assets)
                if preprocessed_data is None:
                    st.stop()
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

else:  # CSV Upload
    st.header("Upload Customer Data (CSV)")
    st.markdown("Upload a CSV file with the required columns. See the [template](#) for reference.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if len(df) > 10000:
            st.error("CSV file is too large. Please upload a file with fewer than 10,000 rows.")
            st.stop()

        st.write("Uploaded Data Sample:")
        st.dataframe(df.head())

        if st.button("Predict on this file"):
            if model_choice == "DistilBERT LLM":
                st.warning("Batch prediction for LLM is slow. Processing in batches for efficiency.")
                predictions = []
                churn_probs = []
                progress_bar = st.progress(0)
                batch_size = 16
                with st.spinner(f"Predicting with {model_choice}..."):
                    tokenizer = assets['tokenizer']
                    model = assets['DistilBERT LLM']
                    for i in range(0, len(df), batch_size):
                        batch_df = df[i:i+batch_size]
                        text_inputs = [convert_to_text(pd.DataFrame([row])) for _, row in batch_df.iterrows()]
                        if None in text_inputs:
                            st.error("Error in text conversion for some rows.")
                            break
                        tokenized_inputs = tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True)
                        with torch.no_grad():
                            logits = model(**tokenized_inputs).logits
                            probs = torch.softmax(logits, dim=1).numpy()
                        predictions.extend(np.argmax(probs, axis=1))
                        churn_probs.extend(probs[:, 1])
                        progress_bar.progress(min((i + batch_size) / len(df), 1.0))

            else:  # Traditional models
                with st.spinner(f"Preprocessing and predicting with {model_choice}..."):
                    preprocessed_df = preprocess_for_traditional_models(df, assets)
                    if preprocessed_df is None:
                        st.stop()
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
