import json

# ---------------------------------------------------------
# CrewAI + Groq compatibility fix
# ---------------------------------------------------------
#
# CrewAI may add a "cache_breakpoint" field to messages.
# Groq does not accept this field.
#
# This disables that marker for our simple Groq application.
# ---------------------------------------------------------

import crewai.llms.cache as crewai_cache

crewai_cache.mark_cache_breakpoint = lambda message: message


# ---------------------------------------------------------
# CrewAI imports
# ---------------------------------------------------------

from crewai import Crew, Process, Task, LLM


# ---------------------------------------------------------
# Our agents
# ---------------------------------------------------------

from agents.requirements_agent import create_requirements_agent
from agents.eligibility_agent import create_eligibility_agent
from agents.recommendation_agent import create_recommendation_agent


# ---------------------------------------------------------
# Load university data
# ---------------------------------------------------------

def load_university_data():

    with open(
        "data/universities.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# Main admission system
# ---------------------------------------------------------

def run_admission_system(student, groq_api_key):

    # -----------------------------------------------------
    # Create Groq LLM
    # -----------------------------------------------------

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0
    )

    # -----------------------------------------------------
    # Load university data
    # -----------------------------------------------------

    university_data = load_university_data()

    university_name = student["university"]
    program_name = student["program"]

    # -----------------------------------------------------
    # Find university
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Find program
    # -----------------------------------------------------

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
    # Create agents
    # -----------------------------------------------------

    requirements_agent = create_requirements_agent(llm)

    eligibility_agent = create_eligibility_agent(llm)

    recommendation_agent = create_recommendation_agent(llm)

    # -----------------------------------------------------
    # TASK 1
    # Requirements
    # -----------------------------------------------------

    requirements_task = Task(

        description=f"""
        Determine the admission requirements for the selected program.

        University:
        {university_name}

        Program:
        {program_name}

        Program data:
        {json.dumps(selected_program, indent=2)}

        Explain:

        1. Minimum percentage
        2. Whether mathematics is required
        3. Whether an entry test is required

        Use ONLY the provided program data.

        Do not invent additional requirements.
        """,

        expected_output="""
        A clear and concise list of the admission requirements.
        """,

        agent=requirements_agent
    )

    # -----------------------------------------------------
    # TASK 2
    # Eligibility
    # -----------------------------------------------------

    eligibility_task = Task(

        description=f"""
        Evaluate the student's eligibility.

        Student Name:
        {student["name"]}

        Qualification:
        {student["qualification"]}

        Percentage:
        {student["percentage"]}

        Mathematics Marks:
        {student["mathematics_marks"]}

        Entry Test Score:
        {student["entry_test_score"]}

        University:
        {university_name}

        Program:
        {program_name}

        Program requirements:
        {json.dumps(selected_program, indent=2)}

        Determine whether the student appears eligible.

        Explain:

        - Requirements satisfied
        - Requirements not satisfied
        - Final eligibility status

        Do not invent requirements.
        """,

        expected_output="""
        A clear eligibility assessment explaining:
        1. Requirements satisfied
        2. Requirements not satisfied
        3. Overall eligibility status
        """,

        agent=eligibility_agent,

        context=[requirements_task]
    )

    # -----------------------------------------------------
    # TASK 3
    # Recommendation
    # -----------------------------------------------------

    recommendation_task = Task(

        description=f"""
        Recommend suitable programs for this student.

        Student Name:
        {student["name"]}

        Qualification:
        {student["qualification"]}

        Percentage:
        {student["percentage"]}

        Mathematics Marks:
        {student["mathematics_marks"]}

        Interest:
        {student["interest"]}

        Available programs:

        {json.dumps(selected_university["programs"], indent=2)}

        Recommend up to 3 programs.

        For each recommendation provide:

        1. Program name
        2. Short explanation why it may suit the student

        Do not guarantee admission.
        """,

        expected_output="""
        Up to three suitable program recommendations
        with a short explanation for each.
        """,

        agent=recommendation_agent,

        context=[
            requirements_task,
            eligibility_task
        ]
    )

    # -----------------------------------------------------
    # Create Crew
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
    # Run Crew
    # -----------------------------------------------------

    result = crew.kickoff()

    return result
