import streamlit as st
import pandas as pd
import os
from PIL import Image
import glob
import joblib
import numpy as np
import pyshark

def main():
    st.set_page_config(page_title="Intrusion Detection Dashboard", layout="wide")

    st.title("🔐 Intrusion Detection System Dashboard")

    # Paths
    results_path = "results/"
    eval_file = os.path.join(results_path, "evaluation_summary.csv")
    conf_matrix_path = os.path.join(results_path, "evaluation_plots/")
    conf_matrix_path_multiclass = os.path.join(results_path, "evaluation_plots_multi/")

    score_plots_path = os.path.join(results_path, "debug_plots/")

    # Load Evaluation Summary
    if os.path.exists(eval_file):
        st.subheader("📊 Model Performance Summary")
        eval_df = pd.read_csv(eval_file)
        st.dataframe(eval_df.style.highlight_max(axis=0), use_container_width=True)
    else:
        st.warning("evaluation_summary.csv not found!")

    # Display Confusion Matrices
    if os.path.exists(conf_matrix_path):
        st.subheader("🧮 Confusion Matrices for multiClass")
        images = sorted(glob.glob(os.path.join(conf_matrix_path, "*.png")))
        cols = st.columns(2)
        for idx, img_path in enumerate(images):
            with cols[idx % 2]:
                st.image(Image.open(img_path), caption=os.path.basename(img_path))
    else:
        st.warning("Confusion matrix images not found.")

    if os.path.exists(conf_matrix_path_multiclass):
        st.subheader("🧮 Confusion Matrices for binary")
        images = sorted(glob.glob(os.path.join(conf_matrix_path_multiclass, "*.png")))
        cols = st.columns(2)
        for idx, img_path in enumerate(images):
            with cols[idx % 2]:
                st.image(Image.open(img_path), caption=os.path.basename(img_path))
    else:
        st.warning("Confusion matrix images not found.")


    # Display Score Distribution Plots (for Unsupervised models)
    if os.path.exists(score_plots_path):
        st.subheader("📈 Score Distribution (Unsupervised Models)")
        images = sorted(glob.glob(os.path.join(score_plots_path, "*.png")))
        cols = st.columns(2)
        for idx, img_path in enumerate(images):
            with cols[idx % 2]:
                st.image(Image.open(img_path), caption=os.path.basename(img_path))
    else:
        st.info("Score distribution plots not found.")



    # Optional: Simulate Live Detection (CSV Stream)
    st.subheader("🚨 Anomaly Detection Simulation")

    uploaded_file = st.file_uploader("Upload recent network log (CSV)", type=["csv"])

    if uploaded_file:
        df_stream = pd.read_csv(uploaded_file)

        score_column = None
        for col in df_stream.columns:
            if 'score' in col.lower() or 'anomaly' in col.lower():
                score_column = col
                break

        if score_column:
            threshold = st.slider("Anomaly Score Threshold", 0.0, 1.0, 0.5)
            anomalies = df_stream[df_stream[score_column] > threshold]

            st.success(f"📌 Total records: {len(df_stream)} | 🚨 Anomalies Detected: {len(anomalies)}")

            st.dataframe(anomalies.head(10))

            if not anomalies.empty:
                st.toast("🚨 Threats detected! Review logs now.", icon="⚠️")
        else:
            st.warning("Couldn't detect an anomaly score column.")

    st.markdown("---")
    st.title("Intrusion Detection - CSV Prediction with Scaling & PCA")

    uploaded_file = st.file_uploader("📄 Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            st.write("📊 Uploaded Data Preview", df.head())

            # Load the scaler, PCA transformer, and RandomForest model
            try:
                scaler = joblib.load("models/scaler_multi.joblib")  # Load the scaler
                pca = joblib.load("models/pca_transformer_multi.joblib")  # Load PCA transformer
                rf_model = joblib.load("models/RandomForest_multi.joblib")  # Load the RandomForest model
                label_encoder = joblib.load("models/label_encoder_multi.joblib")


            except Exception as e:
                st.error(f"Error loading scaler, PCA or model: {e}")
                return

            # Automatically get the columns used during PCA training
            try:
                required_features = list(pca.feature_names_in_)
            except AttributeError:
                st.error("PCA transformer does not contain `feature_names_in_`. Make sure it was trained with a DataFrame.")
                return

            # Check if required features are available in the uploaded file
            

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

            # Step 3: Predict using the RandomForest model
            predictions = rf_model.predict(pca_df)

            # Add the prediction column to the original DataFrame
            df['Prediction'] = predictions

            # Map numeric predictions to class labels
            df['Prediction_Label'] = label_encoder.inverse_transform(predictions)


            st.success("✅ Prediction complete!")
            st.write(df[['Prediction']].head())

            # Provide the option to download the results
            csv = df.to_csv(index=False)
            st.download_button("Download Prediction CSV", csv, file_name="predictions.csv", mime="text/csv")

    



    
            

if __name__ == "__main__":
    main()