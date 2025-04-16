import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Send email alert function
def send_email_alert(subject, message, to_email):
    from_email = "your_email@example.com"  # Your email address
    from_password = "your_email_password"  # Your email password (use app-specific password for Gmail)
    
    # Create the email content
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        # Establish a connection to the SMTP server (e.g., Gmail)
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Encrypt the connection
            server.login(from_email, from_password)
            server.sendmail(from_email, to_email, msg.as_string())
        print(f"Email sent to {to_email} successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


def main():
    st.set_page_config(page_title="🚨 Attack Detector", layout="wide")
    st.title("🚨 Real-Time Attack Classifier")

    uploaded_file = st.file_uploader("📂 Upload network log (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()  # Clean column names
        st.write("📊 Uploaded Data Preview", df.head())

        # Load the scaler, PCA transformer
        try:
            scaler = joblib.load("models/scaler.joblib")  # Load the scaler
            pca = joblib.load("models/pca_transformer.joblib")  # Load PCA transformer
        except Exception as e:
            st.error(f"Error loading scaler or PCA transformer: {e}")
            return

        # Automatically get the columns used during PCA training
        try:
            required_features = list(pca.feature_names_in_)
        except AttributeError:
            st.error("PCA transformer does not contain `feature_names_in_`. Make sure it was trained with a DataFrame.")
            return

        # Check if required features are available in the uploaded file
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded data: {', '.join(missing_cols)}")
            return

        # Extract only the relevant features
        input_data = df[required_features]

        # Step 1: Scale the data using the saved scaler
        input_numeric = input_data.select_dtypes(include=np.number)
        scaled_data = scaler.transform(input_numeric)

        # Convert scaled data back to DataFrame for consistency
        scaled_data_df = pd.DataFrame(scaled_data, columns=input_numeric.columns)

        # Step 2: Apply PCA transformation
        pca_data = pca.transform(scaled_data_df)

        # Create a DataFrame with PCA components
        pca_cols = [f"PC_{i+1}" for i in range(pca_data.shape[1])]
        pca_df = pd.DataFrame(pca_data, columns=pca_cols)

        # Select the model from Streamlit UI
        model_option = st.selectbox("Select the model to use:", ("RandomForest", "XGBoost", "Logistic Regression"))

        # Step 3: Load and use the selected model
        if model_option == "RandomForest":
            try:
                rf_model = joblib.load("models/RandomForest.joblib")  # Load RandomForest model
                predictions = rf_model.predict(pca_df)
            except Exception as e:
                st.error(f"Error loading RandomForest model: {e}")
                return

        elif model_option == "XGBoost":
            try:
                xgb_model = joblib.load("models/XGBoost.joblib")  # Load XGBoost model
                predictions = xgb_model.predict(pca_df)
            except Exception as e:
                st.error(f"Error loading XGBoost model: {e}")
                return

        elif model_option == "Logistic Regression":
            try:
                lr_model = joblib.load("models/LogisticRegression.joblib")  # Load Logistic Regression model
                predictions = lr_model.predict(pca_df)
            except Exception as e:
                st.error(f"Error loading Logistic Regression model: {e}")
                return

        # Add the prediction column to the original DataFrame
        df['Prediction'] = predictions

        st.success("✅ Prediction complete!")
        st.write(df[['Prediction']].head())

        # Explain the prediction using LIME (for attack predictions)
        # if any(predictions == 1):
        #     st.subheader("🔍 Attack Explanation using LIME")

        #     # Create a LIME explainer object
        #     explainer = LimeTabularExplainer(
        #         training_data=pca_df.values,
        #         feature_names=pca_df.columns,
        #         class_names=["Benign", "Attack"],
        #         discretize_continuous=True
        #     )

        #     # Select an instance to explain (you can change this logic to select different data points)
        #     instance_idx = np.where(predictions == 1)[0][0]
        #     instance_to_explain = pca_df.iloc[instance_idx].values.reshape(1, -1)

        #     # Explain the prediction for the instance
        #     explanation = explainer.explain_instance(
        #         instance_to_explain[0], rf_model.predict_proba if model_option != "Logistic Regression" else lr_model.predict_proba, num_features=5
        #     )

        #     # Display the explanation
        #     st.write("Explanation for attack prediction:")
        #     explanation_list = explanation.as_list()
        #     for feature, importance in explanation_list:
        #         st.write(f"Feature: {feature}, Importance: {importance}")

        #     # Optionally plot the explanation (matplotlib)
        #     fig = explanation.as_pyplot_figure()
        #     st.pyplot(fig)

        # Provide the option to download the results
        csv = df.to_csv(index=False)
        st.download_button("Download Prediction CSV", csv, file_name="predictions.csv", mime="text/csv")

        # Display Evaluation Summary CSV at the bottom
        st.subheader("📊 Model Evaluation Summary")
        evaluation_df = pd.read_csv('results/evaluation_summary.csv')

        # Display the evaluation results as a table
        st.dataframe(evaluation_df)

        # Compare models based on accuracy and precision
        st.subheader("🔍 Model Comparison")

        # Extract accuracy and precision values from the evaluation dataframe
        accuracy_model = evaluation_df.loc[evaluation_df['accuracy_binary'].idxmax()]
        precision_model = evaluation_df.loc[evaluation_df['precision'].idxmax()]

        st.write(f"**Best Model for Accuracy:** {accuracy_model['model']} with accuracy: {accuracy_model['accuracy_binary']}")
        st.write(f"**Best Model for Precision:** {precision_model['model']} with precision: {precision_model['precision']}")

        # Conclusion based on comparison
        if accuracy_model['accuracy_binary'] > precision_model['precision']:
            st.write(f"Overall, the **{accuracy_model['model']}** model performs better in terms of accuracy.")
        else:
            st.write(f"Overall, the **{precision_model['model']}** model performs better in terms of precision.")


if __name__ == "__main__":
    main()
