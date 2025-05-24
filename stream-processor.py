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
import plotly.express as px

# Send email alert function
def send_email_alert(subject, message, to_email):
    from_email = "your_email@gmail.com"  # <-- Replace with your Gmail
    from_password = "your_app_password"  # <-- Replace with your App Password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(from_email, from_password)
            server.sendmail(from_email, to_email, msg.as_string())
    except Exception as e:
        st.exception(f"❌ Failed to send email: {e}")

def main():
    st.set_page_config(page_title="🚨 Attack Detector", layout="wide")
    st.title("🚨 Real-Time Attack Classifier")

    uploaded_file = st.file_uploader("📂 Upload network log (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        st.write("📊 Uploaded Data Preview", df.head())

        classification_type = st.selectbox("Select Classification Type", ["Binary", "Multiclass"])

        try:
            if classification_type == "Binary":
                scaler = joblib.load("models/scaler.joblib")
                pca = joblib.load("models/pca_transformer.joblib")
            else:
                scaler = joblib.load("models/scaler_multi.joblib")
                pca = joblib.load("models/pca_transformer_multi.joblib")
                label_encoder = joblib.load("models/label_encoder_multi.joblib")
        except Exception as e:
            st.error(f"Error loading scaler or PCA transformer: {e}")
            return

        try:
            required_features = list(pca.feature_names_in_)
        except AttributeError:
            st.error("PCA transformer lacks `feature_names_in_`.")
            return

        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
            return

        input_data = df[required_features]
        input_numeric = input_data.select_dtypes(include=np.number)
        scaled_data = scaler.transform(input_numeric)
        scaled_data_df = pd.DataFrame(scaled_data, columns=input_numeric.columns)
        pca_data = pca.transform(scaled_data_df)
        pca_cols = [f"PC_{i+1}" for i in range(pca_data.shape[1])]
        pca_df = pd.DataFrame(pca_data, columns=pca_cols)

        model_option = st.selectbox("Select Model", ["RandomForest", "XGBoost", "Logistic Regression"])

        try:
            model_path = f"models/{model_option.replace(' ', '')}"
            if classification_type == "Multiclass":
                model_path += "_multi"
            model_path += ".joblib"

            model = joblib.load(model_path)
            predictions = model.predict(pca_df)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        df['Prediction'] = predictions
        if classification_type == "Multiclass":
            try:
                df['Prediction_Label'] = label_encoder.inverse_transform(predictions)
                st.write(df[['Prediction', 'Prediction_Label']].head())
            except Exception as e:
                st.warning("Error decoding labels: " + str(e))
        else:
            st.write(df[['Prediction']].head())

        st.success("✅ Prediction complete!")

        csv = df.to_csv(index=False)
        st.download_button("Download Prediction CSV", csv, file_name="predictions.csv", mime="text/csv")

        st.subheader("📊 Model Evaluation Summary")

        if classification_type == "Binary":
            try:
                eval_file = 'results/evaluation_summary.csv'
                evaluation_df = pd.read_csv(eval_file)
                st.dataframe(evaluation_df)

                acc_model = evaluation_df.loc[evaluation_df['accuracy_binary'].idxmax()]
                prec_model = evaluation_df.loc[evaluation_df['precision'].idxmax()]

                st.write(f"**Best Accuracy:** {acc_model['model']} - {acc_model['accuracy_binary']:.3f}")
                st.write(f"**Best Precision:** {prec_model['model']} - {prec_model['precision']:.3f}")

                if acc_model['accuracy_binary'] > prec_model['precision']:
                    st.write(f"🎯 **{acc_model['model']}** performs better overall (accuracy).")
                else:
                    st.write(f"🎯 **{prec_model['model']}** performs better overall (precision).")
            except Exception as e:
                st.warning("Evaluation summary file not found or unreadable.")

        else:
            if 'Prediction_Label' in df.columns:
                fig = px.pie(df, names='Prediction_Label', title='Detected Attack Types')
                st.plotly_chart(fig)

                threat_counts = df['Prediction_Label'].value_counts().reset_index()
                threat_counts.columns = ['Attack Type', 'Count']
                fig2 = px.bar(threat_counts, x='Attack Type', y='Count', title="Threat Frequency")
                st.plotly_chart(fig2)
            else:
                st.info("Multiclass evaluation summary is not available yet.")

        

if __name__ == "__main__":
    main()
