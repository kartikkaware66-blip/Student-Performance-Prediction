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
st.markdown("""
<style>
.stApp {
    background-color: #EAF4FF;
}

h1 {
    color: #123B66;
}

h2, h3 {
    color: #1F5F8B;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# STUDENT PERFORMANCE DASHBOARD
# ==============================

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_performance_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict the final result.")

# Student details
student_name = st.text_input("👤 Student Name")
roll_no = st.text_input("🔢 Roll Number")

st.subheader("📚 Student Academic Details")

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
if st.button("🎯 Predict Result"):

    # Input data
    input_data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks],
        "Assignment": [assignment],
        "Internal_Marks": [internal_marks]
    })

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probability = model.predict_proba(input_data)

    pass_probability = probability[0][1] * 100
    fail_probability = probability[0][0] * 100

    st.divider()

    # ==============================
    # RESULT
    # ==============================

    if prediction[0] == 1:
        st.success("✅ Predicted Result: PASS")
    else:
        st.error("❌ Predicted Result: FAIL")

    # ==============================
    # PREDICTION PROBABILITY
    # ==============================

    st.subheader("🎯 Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("PASS Probability", f"{pass_probability:.1f}%")

    with col2:
        st.metric("FAIL Probability", f"{fail_probability:.1f}%")

    # ==============================
    # PERFORMANCE SCORE
    # ==============================

    performance_score = (
        (study_hours / 24 * 100)
        + attendance
        + previous_marks
        + assignment
        + internal_marks
    ) / 5

    if performance_score >= 80:
        performance_level = "🏆 Excellent"
    elif performance_score >= 65:
        performance_level = "👍 Good"
    elif performance_score >= 50:
        performance_level = "⚠️ Average"
    else:
        performance_level = "❗ Poor"

    st.subheader("🏆 Performance Level")
    st.info(performance_level)

    st.write(
        f"**Overall Performance Score:** {performance_score:.1f}%"
    )

    # ==============================
    # INPUT VALUES GRAPH
    # ==============================

    st.subheader("📈 Student Performance Graph")

    graph_data = pd.DataFrame({
        "Parameter": [
            "Study Hours",
            "Attendance",
            "Previous Marks",
            "Assignment",
            "Internal Marks"
        ],
        "Value": [
            study_hours / 24 * 100,
            attendance,
            previous_marks,
            assignment,
            internal_marks
        ]
    })

    st.bar_chart(
        graph_data.set_index("Parameter")
    )

    # ==============================
    # IMPROVEMENT SUGGESTIONS
    # ==============================

    st.subheader("💡 Improvement Suggestions")

    suggestions = []

    if study_hours < 4:
        suggestions.append(
            "📚 Increase your daily study hours."
        )

    if attendance < 75:
        suggestions.append(
            "🏫 Improve your class attendance."
        )

    if previous_marks < 60:
        suggestions.append(
            "📝 Focus more on previous exam topics."
        )

    if assignment < 60:
        suggestions.append(
            "📄 Complete assignments regularly."
        )

    if internal_marks < 60:
        suggestions.append(
            "✍️ Improve preparation for internal exams."
        )

    if len(suggestions) == 0:
        suggestions.append(
            "🌟 Excellent! Keep maintaining your performance."
        )

    for suggestion in suggestions:
        st.write(suggestion)

    # ==============================
    # REPORT
    # ==============================

    report = f"""
STUDENT PERFORMANCE PREDICTION REPORT
======================================

Student Name: {student_name}
Roll Number: {roll_no}

Study Hours: {study_hours}
Attendance: {attendance}%
Previous Marks: {previous_marks}
Assignment Marks: {assignment}
Internal Marks: {internal_marks}

Predicted Result: {"PASS" if prediction[0] == 1 else "FAIL"}

Pass Probability: {pass_probability:.1f}%
Fail Probability: {fail_probability:.1f}%

Performance Score: {performance_score:.1f}%
Performance Level: {performance_level}

======================================
"""

    # Download report
    st.download_button(
        label="📥 Download Prediction Report",
        data=report,
        file_name="student_prediction_report.txt",
        mime="text/plain"
    )