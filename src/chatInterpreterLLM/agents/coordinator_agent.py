from crewai import Agent, LLM
from src.chatInterpreterLLM.env import enviroment

def return_coordinator_agent():
    """Returns an LLM agent for coordinating tasks and delegating to other agents."""
    
    API_KEY = enviroment()

    coordinator_agent = Agent(
        role="General Coordinator",
        goal="Determine if a request or message requires help from an expert agent and delegate it correctly",
        backstory=(
            "You are an intelligent coordinator. You receive user requests and decide whether you can "
            "answer them directly or if you should delegate the task to one of two expert agents: "
            "a AI Model Report Generator LIME or a AI Model Report Generator SHAP." 
            "If the agent to delegate to is the LIME agent, you must:" 
            "1. **Check if the user specifies an instance index** (e.g., Explain row 5 or I want to know why instance 23 was classified that way). " 
            "   - If an index has been specified, Use the tool 'Select Dataset Instance by Index'** to retrieve the requested row from the dataset and store it as `instancia`. "
            "2. **If the user does not specify an index and and the user has provided a new instance (st.session_state.instance_uploaded = True) as a dataframe in the chat, you must use the tool 'Receive New Instance' to process it.** "
            "3. **Once the instance is selected**, immediately **call the 'Generate LIME Explanation' tool** to generate an explanation for that instance." 
            "If the user asks to explain an instance from the dataset and provides an index (e.g. explain instance 5), You must use the corresponding tool that the instance collects and then execute the tool in the explanation." \
            "If you delegate the task to the Lime agent, you must first request the instance to predict and then execute the tool that contains this agent."
            "When using the delegation tool, always ensure that the fields 'task', 'context', and 'coworker' "
            "are provided as plain text strings."
        ),
        allow_delegation=True,
        verbose=True,
        llm=LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature=0.0,
            key=API_KEY
        )
    )
    return coordinator_agent
