import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("student_performance_dataset.csv")

# Display data
print(df.head())

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

plt.figure(figsize=(8, 5))

sns.countplot(x="Result", data=df)

plt.title("Pass vs Fail")
plt.xlabel("Result")
plt.ylabel("Number of Students")

plt.show()

from sklearn.model_selection import train_test_split

# Input features
X = df.drop("Result", axis=1)

# Target
y = df["Result"]

# Convert Pass/Fail into 1/0
y = y.map({"Fail": 0, "Pass": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

# Create model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

print("Model trained successfully!")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Prediction
y_pred = rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# Example student
new_student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [80],
    "Previous_Marks": [75],
    "Assignment": [70],
    "Internal_Marks": [65]
})

prediction = rf_model.predict(new_student)

if prediction[0] == 1:
    print("Predicted Result: PASS")
else:
    print("Predicted Result: FAIL")


import joblib

joblib.dump(rf_model, "student_performance_model.pkl")

print("Model saved successfully!")

model = joblib.load("student_performance_model.pkl")

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")



import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load trained model
model = joblib.load("student_performance_model.pkl")

# Title
st.title("🎓 Student Performance Predictor")
st.write("Enter the student's details to predict the final result.")

# Input fields
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_marks = st.number_input(
    "Previous Marks",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

assignment = st.number_input(
    "Assignment Marks",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

internal_marks = st.number_input(
    "Internal Marks",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

# Prediction button
if st.button("🔮 Predict Result"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks],
        "Assignment": [assignment],
        "Internal_Marks": [internal_marks]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.success("🎉 Predicted Result: PASS")
    else:
        st.error("❌ Predicted Result: FAIL")
