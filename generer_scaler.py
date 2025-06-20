import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Exemple de scaler fictif
scaler = StandardScaler()
scaler.fit(np.random.rand(100, 5))

joblib.dump(scaler, "scaler.pkl")
