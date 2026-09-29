from crewai import Agent


def create_eligibility_agent(llm):
    """
    Creates the Eligibility Agent.
    """

    return Agent(
        role="University Eligibility Evaluator",

        goal=(
            "Evaluate whether the student meets the admission requirements "
            "for the selected university program."
        ),

        backstory=(
            "You are an admission eligibility evaluator. "
            "You compare the student's academic information with the "
            "provided admission requirements. You clearly explain which "
            "requirements are satisfied and which are not satisfied."
        ),

        llm=llm,

        verbose=True,

        allow_delegation=False
    )
