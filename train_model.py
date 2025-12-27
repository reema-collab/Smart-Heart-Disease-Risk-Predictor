import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# ---------- Load Dataset ----------
df = pd.read_csv("heart.csv")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# ---------- Preprocessing ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------- Train Model ----------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------- Predict ----------
y_pred = model.predict(X_test)

# ---------- Evaluate ----------
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# ---------- Save Model & Scaler ----------
pickle.dump(model, open("heart_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\n✔ Model saved as 'heart_model.pkl'")
print("✔ Scaler saved as 'scaler.pkl'")
