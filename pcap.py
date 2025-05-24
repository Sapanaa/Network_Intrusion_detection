import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer
import tempfile
from scapy.all import rdpcap

# --- Utility: Extract basic features from PCAP ---
def extract_features_from_pcap(pcap_path):
    packets = rdpcap(pcap_path)
    data = []

    for pkt in packets:
        if pkt.haslayer("IP"):
            features = {
                "packet_len": len(pkt),
                "proto": pkt["IP"].proto,
                "ttl": pkt["IP"].ttl
            }
            data.append(features)

    df = pd.DataFrame(data)
    df['proto'] = pd.to_numeric(df['proto'], errors='coerce')
    df['ttl'] = pd.to_numeric(df['ttl'], errors='coerce')
    df['packet_len'] = pd.to_numeric(df['packet_len'], errors='coerce')
    df.fillna(0, inplace=True)

    return df


def main():
    st.set_page_config(page_title="🚨 Attack Detector", layout="wide")
    st.title("🚨 Real-Time Attack Classifier")

    uploaded_file = st.file_uploader("📂 Upload network log (CSV or PCAP)", type=["csv", "pcap"])

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1]

        if file_ext == 'csv':
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            st.write("📊 Uploaded CSV Preview", df.head())
        elif file_ext == 'pcap':
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            df = extract_features_from_pcap(tmp_path)
            st.write("📡 Parsed PCAP Data", df.head())
        else:
            st.error("Unsupported file format.")
            return

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
            st.error(f"Missing columns in input: {', '.join(missing_cols)}")
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
                st.warning("Could not load binary evaluation summary.")
        else:
            st.info("Multiclass evaluation summary is not available yet. Only binary results are shown.")


if __name__ == "__main__":
    main()
