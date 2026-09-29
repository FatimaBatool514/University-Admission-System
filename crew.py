import json
import os
# ---------------------------------------------------------
# Fix CrewAI + Groq cache_breakpoint compatibility issue
# ---------------------------------------------------------

import crewai.llms.cache as crewai_cache

crewai_cache.mark_cache_breakpoint = lambda message: message

from crewai import Crew, Process, Task, LLM

from agents.requirements_agent import create_requirements_agent
from agents.eligibility_agent import create_eligibility_agent
from agents.recommendation_agent import create_recommendation_agent


# ---------------------------------------------------------
# 1. Load Groq API key
# ---------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not configured. "
        "Please add it to your Streamlit secrets."
    )


# ---------------------------------------------------------
# 2. Create Groq LLM
# ---------------------------------------------------------

llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0
)


# ---------------------------------------------------------
# 3. Load university data
# ---------------------------------------------------------

def load_university_data():

    with open(
        "data/universities.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# 4. Create the agents
# ---------------------------------------------------------

requirements_agent = create_requirements_agent(llm)

eligibility_agent = create_eligibility_agent(llm)

recommendation_agent = create_recommendation_agent(llm)


# ---------------------------------------------------------
# 5. Create the main function
# ---------------------------------------------------------

def run_admission_system(student):

    university_data = load_university_data()

    university_name = student["university"]
    program_name = student["program"]

    # Find selected university
    selected_university = next(
        (
            university
            for university in university_data["universities"]
            if university["name"] == university_name
        ),
        None
    )

    if not selected_university:
        raise ValueError(
            f"University '{university_name}' was not found."
        )

    # Find selected program
    selected_program = next(
        (
            program
            for program in selected_university["programs"]
            if program["name"] == program_name
        ),
        None
    )

    if not selected_program:
        raise ValueError(
            f"Program '{program_name}' was not found."
        )

    # -----------------------------------------------------
    # TASK 1
    # -----------------------------------------------------

    requirements_task = Task(
        description=f"""
        Determine the admission requirements for this program.

        University:
        {university_name}

        Program:
        {program_name}

        Officially provided program data:
        {json.dumps(selected_program, indent=2)}

        Explain:
        1. Minimum percentage
        2. Whether mathematics is required
        3. Whether an entry test is required

        Do not invent additional requirements.
        """,

        expected_output=(
            "A clear list of the admission requirements "
            "based only on the provided program data."
        ),

        agent=requirements_agent
    )

    # -----------------------------------------------------
    # TASK 2
    # -----------------------------------------------------

    eligibility_task = Task(
        description=f"""
        Evaluate the student's eligibility.

        Student information:

        Name:
        {student["name"]}

        Qualification:
        {student["qualification"]}

        Percentage:
        {student["percentage"]}

        Mathematics marks:
        {student["mathematics_marks"]}

        Entry test score:
        {student["entry_test_score"]}

        Selected university:
        {university_name}

        Selected program:
        {program_name}

        Program requirements:
        {json.dumps(selected_program, indent=2)}

        Determine whether the student appears eligible.

        Clearly explain:
        - Requirements satisfied
        - Requirements not satisfied
        - Final eligibility status

        Do not invent requirements.
        """,

        expected_output=(
            "A clear eligibility assessment explaining "
            "whether the student meets the provided requirements."
        ),

        agent=eligibility_agent,

        context=[requirements_task]
    )

    # -----------------------------------------------------
    # TASK 3
    # -----------------------------------------------------

    recommendation_task = Task(
        description=f"""
        Recommend suitable programs for this student.

        Student:

        Name:
        {student["name"]}

        Qualification:
        {student["qualification"]}

        Percentage:
        {student["percentage"]}

        Mathematics marks:
        {student["mathematics_marks"]}

        Interests:
        {student["interest"]}

        Available programs at the selected university:

        {json.dumps(selected_university["programs"], indent=2)}

        Recommend up to 3 programs.

        For every recommendation provide:
        1. Program name
        2. Why it may suit the student

        Use the student's academic background and interests.
        Do not claim guaranteed admission.
        """,

        expected_output=(
            "A list of up to three suitable programs with "
            "a short explanation for each."
        ),

        agent=recommendation_agent,

        context=[requirements_task, eligibility_task]
    )

    # -----------------------------------------------------
    # CREATE CREW
    # -----------------------------------------------------

    crew = Crew(
        agents=[
            requirements_agent,
            eligibility_agent,
            recommendation_agent
        ],

        tasks=[
            requirements_task,
            eligibility_task,
            recommendation_task
        ],

        process=Process.sequential,

        verbose=True
    )

    # -----------------------------------------------------
    # RUN CREW
    # -----------------------------------------------------

    result = crew.kickoff()

    return result
