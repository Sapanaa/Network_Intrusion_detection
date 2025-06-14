import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from alerts import send_email_alert
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

def detect_autoencoder_anomalies(X, model_dir, threshold=0.01):
    if not TENSORFLOW_AVAILABLE:
        return None
    model_path = os.path.join(model_dir, "Autoencoder.keras")
    try:
        autoencoder = load_model(model_path)
        X_numeric = X.select_dtypes(include=np.number)
        reconstructions = autoencoder.predict(X_numeric)
        mse = np.mean(np.power(X_numeric - reconstructions, 2), axis=1)
        anomalies = mse > threshold
        return pd.DataFrame({"Autoencoder_Anomaly_Score": mse, "Is_Autoencoder_Anomaly": anomalies})
    except Exception as e:
        st.error(f"Error in Autoencoder anomaly detection: {e}")
        return None

def detect_isolationforest_anomalies(X, model_dir):
    model_path = os.path.join(model_dir, "IsolationForest.joblib")
    try:
        iso_forest = joblib.load(model_path)
        X_numeric = X.select_dtypes(include=np.number)
        predictions = iso_forest.predict(X_numeric)  # 1 for normal, -1 for anomaly
        anomalies = predictions == -1
        scores = iso_forest.score_samples(X_numeric)  # Lower scores indicate anomalies
        return pd.DataFrame({"IsolationForest_Anomaly_Score": -scores, "Is_IsolationForest_Anomaly": anomalies})
    except Exception as e:
        st.error(f"Error in IsolationForest anomaly detection: {e}")
        return None
    
def main():
    st.set_page_config(page_title="🚨 Attack Detector", layout="wide")
    st.title("🚨 Real-Time Attack Classifier")

    uploaded_file = st.file_uploader("📂 Upload network log (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        st.write("📊 Uploaded Data Preview", df.head())

        classification_type = st.selectbox("Select Classification Type", ["Binary", "Multiclass"])

        # Load transformers and models
        try:
            if classification_type == "Binary":
                scaler = joblib.load("models/scaler.joblib")
                pca = joblib.load("models/pca_transformer.joblib")
                model_lr = joblib.load("models/LogisticRegression.joblib")
                model_rf = joblib.load("models/RandomForest.joblib")
                model_xgb = joblib.load("models/XGBoost.joblib")
                iso_forest = joblib.load("models/IsolationForest.joblib")
                autoencoder = load_model("models/Autoencoder.keras") if TENSORFLOW_AVAILABLE else None
            else:
                scaler = joblib.load("models/scaler_multi.joblib")
                pca = joblib.load("models/pca_transformer_multi.joblib")
                label_encoder = joblib.load("models/label_encoder_multi.joblib")
                model_lr = joblib.load("models/LogisticRegression_multi.joblib")
                model_rf = joblib.load("models/RandomForest_multi.joblib")
                model_xgb = joblib.load("models/XGBoost_multi.joblib")
                iso_forest = joblib.load("models/IsolationForest_multi.joblib")  # Optional
                autoencoder = load_model("models/Autoencoder_multi.keras") if TENSORFLOW_AVAILABLE else None
        except Exception as e:
            st.error(f"Error loading models or transformers: {e}")
            return

        required_features = list(pca.feature_names_in_)
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
            return

        input_data = df[required_features]
        input_numeric = input_data.select_dtypes(include=np.number)
        if input_numeric.empty:
            st.error("No numeric features available for processing.")
            return

        scaled_data = scaler.transform(input_numeric)
        scaled_data_df = pd.DataFrame(scaled_data, columns=input_numeric.columns)
        pca_data = pca.transform(scaled_data_df)
        pca_cols = [f"PC_{i+1}" for i in range(pca_data.shape[1])]
        pca_df = pd.DataFrame(pca_data, columns=pca_cols)

        model_option = st.selectbox("Select Model", ["RandomForest", "XGBoost", "Logistic Regression"])
        model = {"RandomForest": model_rf, "XGBoost": model_xgb, "Logistic Regression": model_lr}[model_option]

        try:
            predictions = model.predict(pca_df)

            # Anomaly detection
            autoencoder_anomalies = detect_autoencoder_anomalies(df[required_features], "models") if autoencoder else None
            iso_forest_anomalies = detect_isolationforest_anomalies(df[required_features], "models")

            # Combine results
            results_df = df[required_features].copy()
            results_df["Prediction"] = predictions
            if autoencoder_anomalies is not None:
                results_df = results_df.join(autoencoder_anomalies)
            if iso_forest_anomalies is not None:
                results_df = results_df.join(iso_forest_anomalies)

            if classification_type == "Multiclass":
                try:
                    results_df["Prediction_Label"] = label_encoder.inverse_transform(predictions)
                    st.write(results_df[["Prediction", "Prediction_Label", "Autoencoder_Anomaly_Score", "Is_Autoencoder_Anomaly", "IsolationForest_Anomaly_Score", "Is_IsolationForest_Anomaly"]].head())
                except Exception as e:
                    st.warning("Error decoding labels: " + str(e))
            else:
                st.write(results_df[["Prediction", "Autoencoder_Anomaly_Score", "Is_Autoencoder_Anomaly", "IsolationForest_Anomaly_Score", "Is_IsolationForest_Anomaly"]].head())

            # Email alerts for threats
            if classification_type == "Binary":
                threat_rows = results_df[results_df["Prediction"] == 1]
                if not threat_rows.empty:
                    for idx, threat_data in threat_rows.iterrows():
                        send_email_alert(threat_data.to_dict())
                        st.warning(f"Threat detected at index {idx}: {threat_data.to_dict()}")

            # Additional alerts for anomalies
            if autoencoder_anomalies is not None:
                auto_threats = results_df[results_df["Is_Autoencoder_Anomaly"]]
                if not auto_threats.empty:
                    for idx, threat_data in auto_threats.iterrows():
                        st.warning(f"Autoencoder Anomaly detected at index {idx}: Score {threat_data['Autoencoder_Anomaly_Score']:.3f}")
            if iso_forest_anomalies is not None:
                iso_threats = results_df[results_df["Is_IsolationForest_Anomaly"]]
                if not iso_threats.empty:
                    for idx, threat_data in iso_threats.iterrows():
                        st.warning(f"IsolationForest Anomaly detected at index {idx}: Score {threat_data['IsolationForest_Anomaly_Score']:.3f}")

        except Exception as e:
            st.error(f"Error in prediction or anomaly detection: {e}")
            return

        st.success("✅ Prediction complete!")

        csv = results_df.to_csv(index=False)
        st.download_button("Download Prediction CSV", csv, file_name="predictions.csv", mime="text/csv")

        st.subheader("📊 Model Evaluation Summary")
        eval_file = 'results/evaluation_summary.csv' if classification_type == "Binary" else 'results/evaluation_summary_multi.csv'
        try:
            evaluation_df = pd.read_csv(eval_file)
            evaluation_df.columns = evaluation_df.columns.str.strip()
            for col in ['accuracy', 'accuracy_binary', 'precision', 'f1_macro']:
                if col in evaluation_df.columns:
                    evaluation_df[col] = pd.to_numeric(evaluation_df[col], errors='coerce')
            st.dataframe(evaluation_df)

            best_metric = 'accuracy_binary' if classification_type == "Binary" else 'accuracy'
            best_model = evaluation_df.loc[evaluation_df[best_metric].idxmax()] if best_metric in evaluation_df.columns else None
            if best_model is not None:
                st.write(f"**Best {best_metric.replace('_', ' ').capitalize()}:** {best_model['model']} - {best_model[best_metric]:.3f}")
            if classification_type == "Multiclass" and 'f1_macro' in evaluation_df.columns:
                best_f1 = evaluation_df.loc[evaluation_df['f1_macro'].idxmax()]
                st.write(f"**Best F1 Macro:** {best_f1['model']} - {best_f1['f1_macro']:.3f}")
                if best_model and best_f1['f1_macro'] > best_model[best_metric]:
                    st.write(f"🎯 **{best_f1['model']}** performs better overall (F1 Macro).")
                else:
                    st.write(f"🎯 **{best_model['model']}** performs better overall (Accuracy).")
            elif best_model:
                st.write(f"🎯 **{best_model['model']}** performs best overall.")
        except FileNotFoundError:
            st.warning(f"{eval_file} not found.")
        except Exception as e:
            st.error(f"Failed to load evaluation summary: {e}")

if __name__ == "__main__":
    main()