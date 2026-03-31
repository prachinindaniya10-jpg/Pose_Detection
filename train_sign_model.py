import pickle
from sklearn.ensemble import RandomForestClassifier

# Load A data
with open("A_data.pickle", "rb") as f:
    data_A, labels_A = pickle.load(f)

# Load B data
with open("B_data.pickle", "rb") as f:
    data_B, labels_B = pickle.load(f)

# Combine data
data = data_A + data_B
labels = labels_A + labels_B

# Train model
model = RandomForestClassifier()
model.fit(data, labels)

# Save model
with open("model.p", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")