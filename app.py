import streamlit as st
import logging
from autogen import ConversableAgent, RetrieveUserProxyAgent, AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up the title of the Streamlit app
st.title("Meal Plan Assistant")

# Load OpenAI API key and Edamam API key from Streamlit secrets
OPEN_API_KEY = st.secrets["OPENAI_API_KEY"]
EDAMAM_APP_KEY = st.secrets["EDAMAM_APP_KEY"]
EDAMAM_APP_ID = "fcbbd9b3"

config_list = [{"model": "gpt-3.5-turbo", "api_key": OPEN_API_KEY}]

# Define the Edamam API agent to retrieve recipes
class EdamamAPIAgent(AssistantAgent):
    def __init__(self, name, app_id, app_key, llm_config):
        super().__init__(name, llm_config=llm_config)
        self.app_id = app_id
        self.app_key = app_key

    def search_recipes(self, query, health_labels=None, diet_labels=None, max_results=10):
        endpoint = "https://api.edamam.com/api/recipes/v2"
        params = {
            "type": "public",
            "q": query,
            "app_id": self.app_id,
            "app_key": self.app_key,
            "to": max_results
        }
        if health_labels:
            params["health"] = health_labels
        if diet_labels:
            params["diet"] = diet_labels
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return {
                "source": "Edamam API",
                "data": response.json()
            }
        except requests.exceptions.RequestException as e:
            return {"source": "Edamam API", "error": f"Error: {str(e)}", "data": None}

# Create the Edamam agent
edamam_agent = EdamamAPIAgent("EdamamAgent", EDAMAM_APP_ID, EDAMAM_APP_KEY, llm_config={"config_list": config_list})

# Create the main assistant agent for meal planning
assistant = AssistantAgent(
    name="MealPlanAssistant",
    system_message='''You are a helpful meal planning assistant. Greet the user, ask for their information (name, zip, chronic disease, cuisine preference and ingredient dislike). Tailor the meal plan based on the customer's chronic disease and preferences, and use the Edamam API to find specific recipes that match the customer's needs.''',
    llm_config={"config_list": config_list}
)

# Create a RetrieveUserProxyAgent for document-based retrieval
def label_rag_response(response):
    return {"source": "RAG System (PDFs)", "data": response}

ragproxyagent = RetrieveUserProxyAgent(
    name="UserProxy",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=1,
    retrieve_config={
        "task": "qa",
        "docs_path": [
            "https://www.nhlbi.nih.gov/sites/default/files/publications/WeekOnDASH.pdf",
            "https://www.nhlbi.nih.gov/files/docs/public/heart/new_dash.pdf",
        ],
        "chunk_token_size": 1000,
        "model": config_list[0]["model"],
        "chunk_mode": "multi_lines",
        "custom_callback": label_rag_response
    },
    llm_config={"config_list": config_list},
    function_map={"search_recipes": edamam_agent.search_recipes}
)

# Function to initiate the chat
def start_chat(user_message):
    # Initial greeting message or user-provided message
    if not user_message:
        user_message = '''Hello! I'm your meal planning assistant. I'll ask you a few questions to create a personalized meal plan. What's your name?'''
    
    # Start or continue chat
    ragproxyagent.initiate_chat(
        assistant,
        message=user_message
    )

# Session state to store the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input and process it
user_input = st.chat_input("You: ")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process the response and initiate the chat
    start_chat(user_input)
    
    # Assuming the assistant's response is retrieved here
    # Simulating the response
    assistant_response = "This is a sample response based on the Edamam API and RAG system."
    
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

# Button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()
