import json

import streamlit as st

from crew import run_admission_system


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI University Admission Advisor",
    page_icon="🎓",
    layout="centered"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("🎓 AI University Admission Advisor")

st.write(
    "Enter your academic information and let our "
    "multi-agent system evaluate your admission profile."
)


# ---------------------------------------------------------
# Load university data
# ---------------------------------------------------------

def load_data():

    with open(
        "data/universities.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


data = load_data()

universities = data["universities"]


# ---------------------------------------------------------
# Student information
# ---------------------------------------------------------

st.header("👨‍🎓 Student Information")


student_name = st.text_input(
    "Student Name"
)


qualification = st.selectbox(
    "Qualification",
    [
        "FSc Pre-Engineering",
        "FSc Pre-Medical",
        "ICS",
        "A-Levels",
        "Intermediate",
        "Other"
    ]
)


percentage = st.number_input(
    "Overall Percentage",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)


mathematics_marks = st.number_input(
    "Mathematics Marks (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)


entry_test_score = st.number_input(
    "Entry Test Score (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)


interest = st.selectbox(
    "Area of Interest",
    [
        "Computer Science",
        "Artificial Intelligence",
        "Data Science",
        "Software Engineering",
        "Business Administration"
    ]
)


# ---------------------------------------------------------
# University selection
# ---------------------------------------------------------

st.header("🏫 University & Program")


university_names = [
    university["name"]
    for university in universities
]


selected_university = st.selectbox(
    "Select University",
    university_names
)


# Find selected university
university = next(
    university
    for university in universities
    if university["name"] == selected_university
)


program_names = [
    program["name"]
    for program in university["programs"]
]


selected_program = st.selectbox(
    "Select Program",
    program_names
)


# ---------------------------------------------------------
# Analyze button
# ---------------------------------------------------------

if st.button(
    "🔍 Check Admission",
    type="primary"
):

    if not student_name.strip():

        st.warning(
            "Please enter the student's name."
        )

        st.stop()


    # Create student profile

    student = {

        "name": student_name,

        "qualification": qualification,

        "percentage": percentage,

        "mathematics_marks": mathematics_marks,

        "entry_test_score": entry_test_score,

        "interest": interest,

        "university": selected_university,

        "program": selected_program
    }


    # -----------------------------------------------------
    # Run CrewAI
    # -----------------------------------------------------

    with st.spinner(
        "🤖 Our AI agents are analyzing your admission profile..."
    ):

        try:

            result = run_admission_system(
                student
            )

        except Exception as e:

            st.error(
                f"An error occurred: {str(e)}"
            )

            st.stop()


    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    st.success(
        "Admission analysis completed!"
    )


    st.header("📋 Admission Assessment")


    st.write(result)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "This application is an educational demonstration. "
    "Admission requirements should always be verified "
    "with the university's official admission office."
)

groq_api_key = st.secrets["GROQ_API_KEY"]

result = run_admission_system(
    student,
    groq_api_key
)
