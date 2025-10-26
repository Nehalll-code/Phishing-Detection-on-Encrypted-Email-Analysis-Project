# prepare_model.py
import joblib
import numpy as np

# Parameters
MODEL_PATH = "baseline_model.pkl"
SCALE = 10_000  # fixed-point scaling factor. Increase for more precision.

# Load model
model = joblib.load(MODEL_PATH)
# Ensure model has coef_ and intercept_
coef = model.coef_.ravel()  # shape (n_features,)
intercept = float(model.intercept_.ravel()[0])

# Scale and round to integers
coef_scaled = np.round(coef * SCALE).astype("int64")
intercept_scaled = int(round(intercept * SCALE))

# Save
np.save("coef_scaled.npy", coef_scaled)
np.save("intercept_scaled.npy", np.array([intercept_scaled], dtype="int64"))

print("Saved coef_scaled.npy and intercept_scaled.npy with SCALE =", SCALE)
