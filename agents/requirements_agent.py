from crewai import Agent


def create_requirements_agent(llm):
    """
    Creates the Admission Requirements Agent.
    """

    return Agent(
        role="University Admission Requirements Specialist",

        goal=(
            "Identify and clearly explain the admission requirements "
            "for the student's selected university and program."
        ),

        backstory=(
            "You are an admission requirements specialist. "
            "You carefully examine the university and program information "
            "provided to you. You explain requirements clearly and never "
            "invent information that is not provided."
        ),

        llm=llm,

        verbose=True,

        allow_delegation=False
    )
