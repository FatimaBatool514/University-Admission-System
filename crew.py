GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0
)
Requirements Agent
Eligibility Agent
Recommendation Agent
Requirements Task
      ↓
Eligibility Task
      ↓
Recommendation Task
crew.kickoff()
