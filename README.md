
# ❤️ Smart Health Risk Predictor using Machine Learning

A Machine Learning based web application that predicts **Heart Disease risk** based on medical inputs such as age, cholesterol level, blood pressure, and more.  
The system provides an early risk estimation to help users and healthcare professionals take preventive measures.

---

## 🚀 Features
- Predicts **heart disease risk** using a trained ML model
- Interactive Web UI built using **Streamlit**
- Real-time prediction with **user inputs**
- **88.52% Model Accuracy**
- Saves trained model for reusability

---

## 🧠 Technologies Used

| Component | Technology |
|----------|------------|
| Programming Language | Python |
| Machine Learning Model | Random Forest Classifier |
| ML Libraries | Pandas, NumPy, Scikit-Learn |
| Visualization | Matplotlib, Seaborn |
| Web UI | Streamlit |
| Model Saving | Pickle |

---

## 📂 Project Structure

SmartHealthPredictor/
│
├── heart.csv            # Dataset
├── eda.py               # Data exploration & visualization
├── train_model.py       # Model training and saving
├── app.py               # Streamlit Web App
├── heart_model.pkl      # Saved ML model
└── scaler.pkl           # Saved data scaler

---

## 🗄 Dataset

Dataset used: **Heart Disease UCI Dataset**  
Source: https://www.kaggle.com/datasets/ronitf/heart-disease-uci

The dataset contains **303 rows & 14 columns**, with the following target:
- `1` → Heart Disease present  
- `0` → No Heart Disease

---

## 🔍 Exploratory Data Analysis (EDA)

- No missing values in the dataset
- Balanced dataset: Target 0 (138 samples), Target 1 (165 samples)
- Strong influencing features: `cp`, `thalach`, `oldpeak`, `ca`

---

## 🏋️‍♂️ Model Training

We trained a **Random Forest Classifier** due to its robustness with small medical datasets.

### 👇 Training Code
```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### 🎯 Output
```
Model Accuracy: 88.52%
```

The trained model & scaler are saved using `pickle`:
```python
pickle.dump(model, open("heart_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
```

---

## 🌐 Running the Web App

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py
```

### Access in browser
```
http://localhost:8501
```

---

## 🖥️ User Interface

The app allows users to enter medical values and predicts risk:

```python
if prediction == 1:
    st.error("⚠ High Risk of Heart Disease!")
else:
    st.success("✅ Low Risk of Heart Disease.")
```

---

## 📌 Results

| Input Example | Output |
|--------------|--------|
| Age: 55, BP: 160, Chol: 280, MaxHR:120 | ⚠ High Risk |
| Age: 22, BP:110, Chol:180, MaxHR:180 | ✅ Low Risk |

---

## 📝 Future Enhancements
- Add **Diabetes & Kidney Disease prediction**
- Deploy to **Streamlit Cloud / HuggingFace**
- Add **Flutter Mobile App**
- Integrate **SHAP Explainability**
- Add **cloud database logging**

---

## 🙋 Author
**Reema Basheer**

---

## ⭐ Support
If you like this project, give it a ⭐ on GitHub!
