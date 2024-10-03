#from langchain.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field

from typing import Annotated, TypedDict, Literal, Optional
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
import requests
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import datetime

# Load environment variables from .env file
load_dotenv()

# Scout settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

# LangSmith Tracing
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "scout-agent"

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "scout"
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# DEFINE STRUCTURED OUTPUT
class FinalResponse(BaseModel):
    """
    Response to the user
    """
    answer: str = Field(description="The summary of your answer")
    source: list = Field(description="A list of urls or trail ids")

def create_trail_iframe(trail_id: str) -> str:
    """Create an iframe HTML code for a given trail ID"""
    return f"""
    <iframe width="100%" height="350" frameborder="0" 
            src="https://www.trailforks.com/widgets/trail/?trailid={trail_id}&w=100%&h=350px&activitytype=6&map=1&basemap=trailforks&elevation=1&photos=0&title=1&info=1&trail_opacity=25&v=2&basicmap=1">
    </iframe>
    """


# DEFINE INPUT, OUTPUT AND STATE
class AgentInput(MessagesState):
    pass


class AgentOutput(TypedDict):
    # Final structured response from the agent
    final_response: FinalResponse


class AgentState(MessagesState):
    # Final structured response from the agent
    final_response: FinalResponse


# Tools
@tool
def search_trailforks(trail_name: str) -> dict:
    """
    Use this tool to get id for a specific trail or route based on input trail_name
    """
    app_id = 76
    app_secret = '8c1cea65e4305c3f'

    url = f"https://www.trailforks.com/api/1/search?term={trail_name}&type=trail&scope=full&app_id={app_id}&app_secret={app_secret}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Extracting trail ID from the response
        if data:
            id = data['data']['hits']['hits'][0]['_id']  # return the top result out of 50
            # source = data['data']['hits']['hits'][0]['_source']
            return {'answer': f"Trail Map Preview", 'source': [id]}
        return {'answer': "No trail found", 'source': []}


def retrieve_pinecone(query: str) -> dict:
    """
    Retrieve relevant context and source from Pinecone
    - query: user's question
    - We filter out score < 0.8 to ensure the relevance of the context
    """
    context = []
    source = []
    added_urls = set()
    separator = "\n"
    vectorstore = vector_store

    # Find top-k relevant articles from Pinecone
    results = vectorstore.similarity_search_with_score(query, k=4)

    for res in results:
        x, score = res
        if score > 0.83:
            context.append(separator + x.page_content)
            metadata = {key: value.date().isoformat() if isinstance(value,
                                                                    datetime.datetime) else value
                        for key, value in x.metadata.items()
                        if key in ['item_source', 'item_title', 'item_url',
                                   'item_image_url', 'item_created_at']}
            # Check if we have the same source url
            if metadata['item_url'] not in added_urls:
                source.append(metadata)
                added_urls.add(metadata['item_url'])

    context = "".join(context)

    # Return structured data
    return {'answer': context, 'source': source}

model = ChatOpenAI(model="gpt-4o-mini")
tools = [retrieve_pinecone, search_trailforks, FinalResponse]

model_with_response_tool = model.bind_tools(tools, tool_choice="any")


# GRAPH
def call_model(state: AgentState):
    response = model_with_response_tool.invoke(state['messages'])
    return {"messages": response}


def respond(state: AgentState):
    tool_args = state['messages'][-1].tool_calls[0]['args']

    # Create a dictionary from tool_args
    response_dict = dict(tool_args)

    # Check if any of the sources is a trail ID (assuming it's numeric)
    iframe = None
    if 'source' in response_dict and response_dict['source']:
        for source in response_dict['source']:
            if isinstance(source, str) and source.isdigit():
                iframe = create_trail_iframe(source)
                break

    # Add iframe to response_dict if we created one
    if iframe:
        response_dict['iframe'] = iframe

    # Create FinalResponse object
    response = FinalResponse(**response_dict)
    return {"final_response": response}
    # Original ===
    # response = FinalResponse(**state['messages'][-1].tool_calls[0]['args'])
    # return {"final_response": response}
    #=======


# ROUTING FUNCTION
def should_continue(state: AgentState):
    message = state['messages']
    last_message = message[-1]
    if len(last_message.tool_calls)==1 and last_message.tool_calls[0]['name'] == 'FinalResponse':
        return "respond"
    else:
        return "continue"


# GRAPH STRUCTURE
workflow = StateGraph(AgentState, input=AgentInput, output=AgentOutput)
workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", ToolNode(tools))

# Start Node
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "respond": "respond",
    },
)
workflow.add_edge("tools", "agent")
workflow.add_edge("respond", END)
graph = workflow.compile()