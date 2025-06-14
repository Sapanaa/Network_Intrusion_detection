import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from alerts import send_email_alert

def main():
    st.set_page_config(page_title="🚨 Attack Detector", layout="wide")
    st.title("🚨 Real-Time Attack Classifier")

    uploaded_file = st.file_uploader("📂 Upload network log (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        st.write("📊 Uploaded Data Preview", df.head())

        classification_type = st.selectbox("Select Classification Type", ["Binary", "Multiclass"])

        # Load transformers
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
            st.error("PCA transformer lacks `feature_names_in_`. Please retrain using scikit-learn >=1.0.")
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

            #yo add gareko
            # Assuming for binary classification: 1 means threat, 0 means no threat
            if classification_type == "Binary":
                threat_rows = df.iloc[predictions == 1]  # select rows where prediction == 1 (threat)
                if not threat_rows.empty:
                    for idx, threat_data in threat_rows.iterrows():
                        send_email_alert(threat_data.to_dict())

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
            eval_file = 'results/evaluation_summary.csv'
            try:
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
                st.warning(f"Could not load binary evaluation file: {e}")

        else:
            eval_file = 'results/evaluation_summary_multi.csv'
            try:
                evaluation_df = pd.read_csv(eval_file)
                evaluation_df.columns = evaluation_df.columns.str.strip()

                # Convert metric columns safely
                for col in ['accuracy', 'accuracy_weighted', 'f1_macro']:
                    if col in evaluation_df.columns:
                        evaluation_df[col] = pd.to_numeric(evaluation_df[col], errors='coerce')

                st.dataframe(evaluation_df)

                # Get best performing models
                best_accuracy = evaluation_df.loc[evaluation_df['accuracy'].idxmax()] if 'accuracy' in evaluation_df.columns else None
                best_f1 = evaluation_df.loc[evaluation_df['f1_macro'].idxmax()] if 'f1_macro' in evaluation_df.columns else None

                if best_accuracy is not None:
                    st.write(f"**Best Accuracy:** {best_accuracy['model']} - {best_accuracy['accuracy']:.3f}")
                if best_f1 is not None:
                    st.write(f"**Best F1 Macro:** {best_f1['model']} - {best_f1['f1_macro']:.3f}")

                if best_accuracy is not None and best_f1 is not None:
                    if best_accuracy['accuracy'] >= best_f1['f1_macro']:
                        st.write(f"🎯 **{best_accuracy['model']}** performs better overall (accuracy).")
                    else:
                        st.write(f"🎯 **{best_f1['model']}** performs better overall (macro F1 score).")
            except FileNotFoundError:
                st.warning("Multiclass evaluation summary file not found.")
            except Exception as e:
                st.error(f"Failed to load or parse evaluation summary: {e}")

if __name__ == "__main__":
    main()
