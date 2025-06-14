import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scapy.utils import rdpcap
from scapy.layers.inet import IP, TCP
from alerts import send_email_alert

# Global dictionary to store flow data from PCAP
flow_data = {}

def process_pcap(pcap_file):
    """
    Process the PCAP file to extract flow-based features.
    """
    packets = rdpcap(pcap_file)
    for pkt in packets:
        try:
            if IP in pkt:
                ip_layer = pkt[IP]
                flow_key = f"{ip_layer.src}_{ip_layer.dst}_{pkt.get('TCP', {}).get('sport', 0)}_{pkt.get('TCP', {}).get('dport', 0)}"
                if flow_key not in flow_data:
                    flow_data[flow_key] = {
                        'packets': 0,
                        'lengths': [],
                        'start_time': float(pkt.time),
                        'end_time': float(pkt.time)
                    }
                flow = flow_data[flow_key]
                flow['packets'] += 1
                flow['lengths'].append(len(pkt))
                flow['end_time'] = float(pkt.time)
                if TCP in pkt:
                    tcp_layer = pkt[TCP]
                    flow['tcp_flags'] = tcp_layer.flags
                print(f"Processed flow: {flow_key}, Packets: {flow['packets']}")  # Debug print
        except Exception as e:
            st.warning(f"Packet processing error: {e}")

def main():
    """
    Main function to run the Streamlit app for PCAP-based attack classification.
    """
    st.set_page_config(page_title="🚨 Attack Detector", layout="wide")
    st.title("🚨 PCAP Attack Classifier")

    # File uploader for PCAP
    uploaded_file = st.file_uploader("📂 Upload PCAP file", type=["pcap", "pcapng"])
    if uploaded_file is not None:
        st.write("📊 Processing PCAP file...")
        # Save uploaded file temporarily
        with open("temp.pcap", "wb") as f:
            f.write(uploaded_file.getbuffer())
        process_pcap("temp.pcap")
        st.success("PCAP file processed successfully!")

        # Load transformers and models
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
            st.error(f"Error loading scaler or PCA: {e}")
            return

        required_features = list(pca.feature_names_in_) if hasattr(pca, 'feature_names_in_') else []
        if not required_features:
            st.error("PCA transformer lacks feature names. Please retrain with scikit-learn >=1.0.")
            return
        st.write("Required features:", required_features)  # Debug print

        # Process flows into features
        batch_size = 50
        batch_features = []
        for flow_key, flow in flow_data.items():
            if flow['packets'] > 0:
                features = {
                    "protocol": 6,  # Default to TCP
                    "packet_length": np.mean(flow['lengths']) if flow['lengths'] else 0,
                    "tcp_flag_syn": int((flow.get('tcp_flags', 0) & 0x02) > 0) if flow.get('tcp_flags') else 0,
                    "tcp_flag_ack": int((flow.get('tcp_flags', 0) & 0x10) > 0) if flow.get('tcp_flags') else 0
                }
                # Pad with zeros for any missing required features
                for feat in required_features:
                    if feat not in features:
                        features[feat] = 0
                batch_features.append(features)
                if len(batch_features) >= batch_size:
                    break

        if batch_features:
            df_features = pd.DataFrame(batch_features)
            st.write(f"Processed {len(df_features)} flows from PCAP")

            # Align with required features
            input_data = df_features.reindex(columns=required_features, fill_value=0)
            input_numeric = input_data.select_dtypes(include=np.number)
            if input_numeric.empty:
                st.error("No numeric features available for scaling.")
                return
            scaled_data = scaler.transform(input_numeric)
            pca_data = pca.transform(scaled_data)
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

            # Display results
            results_df = pd.DataFrame({
                "Prediction": predictions,
                "Source IP": [k.split('_')[0] for k in flow_data.keys()],
                "Destination IP": [k.split('_')[1] for k in flow_data.keys()]
            })
            st.dataframe(results_df)

            # Send email alerts for threats
            if classification_type == "Binary" and 1 in predictions:
                threat_rows = results_df[results_df['Prediction'] == 1]
                for idx, threat_data in threat_rows.iterrows():
                    send_email_alert({
                        "Source IP": threat_data["Source IP"],
                        "Destination IP": threat_data["Destination IP"],
                        "Prediction": "Threat Detected"
                    })
                    st.warning(f"Threat detected: {threat_data['Source IP']} -> {threat_data['Destination IP']}")
        else:
            st.write("No flows extracted from PCAP. Check file content.")
    else:
        st.write("Please upload a PCAP file to analyze.")

    st.success("✅ Prediction complete!")

if __name__ == "__main__":
    main()