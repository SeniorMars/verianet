"""Export weights to JSON for web demo"""
import numpy as np
import json

# Load weights
data = np.load("verysmallnn_weights.npz")

# Export to JSON (we'll add sample images to the HTML directly)
weights = {
    "W1": data["W1"].tolist(),
    "W2": data["W2"].tolist(),
    "W3": data["W3"].tolist(),
    "b1": data["b1"].tolist(),
    "b2": data["b2"].tolist(),
    "b3": data["b3"].tolist(),
    "architecture": {
        "input": int(data["W1"].shape[0]),
        "hidden1": int(data["W1"].shape[1]),
        "hidden2": int(data["W2"].shape[1]),
        "output": int(data["W3"].shape[1])
    }
}

with open("demo/weights.json", "w") as f:
    json.dump(weights, f)

print("Exported to demo/weights.json")
print(f"Architecture: {weights['architecture']}")
