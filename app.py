import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }

    .header {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }

    .header h1 {
        color: white;
        margin-bottom: 5px;
    }

    .header p {
        color: #e5e7eb;
        font-size: 18px;
    }

    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    .result-pass {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #dcfce7;
        border: 2px solid #22c55e;
    }

    .result-fail {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #fee2e2;
        border: 2px solid #ef4444;
    }

    .metric-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f1f5f9;
        text-align: center;
        border: 1px solid #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("student_performance_model.pkl")

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <h1>🎓 Student Performance Prediction</h1>
    <p>AI-Based Machine Learning Prediction System</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Enter the student's academic information below to predict "
    "their performance."
)

# ---------------- STUDENT INFORMATION ----------------
st.markdown("### 👤 Student Information")

col1, col2 = st.columns(2)

with col1:
    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student name"
    )

with col2:
    roll_no = st.text_input(
        "Roll Number",
        placeholder="Enter roll number"
    )

# ---------------- ACADEMIC DETAILS ----------------
st.markdown("### 📚 Academic Details")

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.number_input(
        "📖 Study Hours / Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

with col2:
    attendance = st.number_input(
        "🏫 Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

with col3:
    previous_marks = st.number_input(
        "📝 Previous Marks",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

col4, col5 = st.columns(2)

with col4:
    assignment = st.number_input(
        "📚 Assignment Marks",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

with col5:
    internal_marks = st.number_input(
        "✍️ Internal Marks",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

# ---------------- PREDICT BUTTON ----------------
st.markdown("---")

predict = st.button(
    "🎯 Predict Student Performance",
    use_container_width=True
)

# ---------------- PREDICTION ----------------
if predict:

    if not student_name or not roll_no:
        st.warning("⚠️ Please enter Student Name and Roll Number.")

    else:

        input_data = pd.DataFrame({
            "Study_Hours": [study_hours],
            "Attendance": [attendance],
            "Previous_Marks": [previous_marks],
            "Assignment": [assignment],
            "Internal_Marks": [internal_marks]
        })

        prediction = model.predict(input_data)
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

        # ---------------- RESULT ----------------
        st.markdown("---")
        st.markdown("## 📊 Prediction Result")

        if prediction[0] == 1:
            st.markdown("""
            <div class="result-pass">
                <h1>✅ PASS</h1>
                <h3>Student is predicted to PASS</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-fail">
                <h1>❌ FAIL</h1>
                <h3>Student is predicted to FAIL</h3>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # ---------------- METRICS ----------------
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Student",
                student_name
            )

        with col2:
            st.metric(
                "Roll Number",
                roll_no
            )

        with col3:
            st.metric(
                "Pass Probability",
                f"{pass_probability:.1f}%"
            )

        with col4:
            st.metric(
                "Performance",
                performance
            )

        # ---------------- PROBABILITY ----------------
        st.markdown("### 🎯 Prediction Probability")

        st.write(
            f"✅ Pass Probability: **{pass_probability:.2f}%**"
        )
        st.progress(int(pass_probability))

        st.write(
            f"❌ Fail Probability: **{fail_probability:.2f}%**"
        )

        # ---------------- PERFORMANCE CHART ----------------
        st.markdown("### 📈 Academic Performance")

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

        st.bar_chart(
            chart_data.set_index("Parameter")
        )

        # ---------------- SUGGESTIONS ----------------
        st