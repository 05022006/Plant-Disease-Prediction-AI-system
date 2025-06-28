import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load data
df = pd.read_csv(r"C:\Users\LINGESH\Desktop\plant disease app\synthetic_plant_signal_dataset.csv")

# Split features and labels
X = df[['mean_voltage', 'std_deviation', 'peak_count', 'fft_energy']]
y = df['disease_label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, r"C:\Users\LINGESH\Desktop\plant disease app\trained_model.pkl")
joblib.dump(scaler, r"C:\Users\LINGESH\Desktop\plant disease app\scaler.pkl")
