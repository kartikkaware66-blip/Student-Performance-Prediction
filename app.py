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
    color: #38DF8;
}

h2, h3 {
    color: #E2E8F0;
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

    if student_name == "" or roll_no == "":
        st.warning("⚠️ Please enter Student Name and Roll Number")

    else:
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
        probability = model.predict_proba(input_data)[0]

        fail_probability = probability[0] * 100
        pass_probability = probability[1] * 100

        # Performance Level
        average = (
            attendance +
            previous_marks +
            assignment +
            internal_marks
        ) / 4

        if average >= 85:
            performance = "Excellent 🏆"
        elif average >= 70:
            performance = "Good 👍"
        elif average >= 50:
            performance = "Average 🙂"
        else:
            performance = "Poor ⚠️"

        # Student Report
        st.markdown("---")
        st.subheader("📊 Student Performance Report")

        st.write("👤 **Student Name:**", student_name)
        st.write("🎫 **Roll Number:**", roll_no)

        if prediction[0] == 1:
            st.success("🎉 Predicted Result: PASS")
        else:
            st.error("❌ Predicted Result: FAIL")

        st.write(f"🎯 **Pass Probability:** {pass_probability:.2f}%")
        st.write(f"❌ **Fail Probability:** {fail_probability:.2f}%")
        st.write(f"🏆 **Performance Level:** {performance}")

        st.write("📈 Pass Probability")
        st.progress(int(pass_probability))
        # 📈 Student Performance Dashboard
st.markdown("---")
st.subheader("📈 Student Performance Dashboard")

chart_data = pd.DataFrame({
    "Parameter": [
        "Study Hours",
        "Attendance",
        "Previous Marks",
        "Assignment",
        "Internal Marks"
    ],
    "Score": [
        study_hours,
        attendance,
        previous_marks,
        assignment,
        internal_marks
    ]
})

st.bar_chart(chart_data.set_index("Parameter"))
        # Improvement Suggestions
st.markdown("---")
st.subheader("💡 Improvement Suggestions")

suggestions = []

if study_hours < 3:
    suggestions.append("📚 Increase study hours to at least 3 hours daily.")

if attendance < 75:
    suggestions.append("🏫 Improve attendance. Try to maintain above 75%.")

if previous_marks < 50:
    suggestions.append("📝 Focus more on previous weak subjects.")

if assignment < 50:
    suggestions.append("📖 Complete assignments regularly.")

if internal_marks < 50:
    suggestions.append("✍️ Improve internal test preparation.")

if not suggestions:
    st.success("🌟 Excellent! Keep maintaining your current performance.")

else:
    for suggestion in suggestions:
        st.info(suggestion)