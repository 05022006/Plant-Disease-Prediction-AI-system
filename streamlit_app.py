import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load(r"C:\Users\LINGESH\Desktop\plant disease app\trained_model.pkl")
scaler = joblib.load(r"C:\Users\LINGESH\Desktop\plant disease app\scaler.pkl")

# Streamlit page setup
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Prediction from Electrical Signals")
st.write("Upload a CSV file")

# File uploader (user uploads new plant signal data)
uploaded_file = st.file_uploader("Upload your plant signal CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load uploaded CSV
        df = pd.read_csv(uploaded_file)

        # Extract only the features needed for prediction
        features = df[['mean_voltage', 'std_deviation', 'peak_count', 'fft_energy']]
        scaled_features = scaler.transform(features)

        # Make predictions
        predictions = model.predict(scaled_features)

        # Create a results DataFrame showing only predictions
        results_df = pd.DataFrame({'Predicted Disease': predictions})

        # Display results
        st.success("✅ Prediction of each Plant Disease is completed!")
        st.dataframe(results_df)

        # Offer download of prediction results
        st.download_button("📥 Download Prediction Results",
                           data=results_df.to_csv(index=False),
                           file_name="predictions.csv",
                           mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
        st.info("Make sure your CSV includes only: mean_voltage, std_deviation, peak_count, fft_energy")
