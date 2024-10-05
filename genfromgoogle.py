from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.prompts import PromptTemplate
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()


def getGoogleAgent(model="gpt-3.5-turbo", chat_history=''):
    print(chat_history)
    LLM = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=model)
    google_search = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
    search_tool = Tool(
        name="Google Search",
        func=google_search.run,
        description="Useful for when you need to answer questions with search",
    )
    datetimetool = Tool(
    name="datetimetool",
    func=lambda x: datetime.now().strftime('%A %d %B %Y, %I:%M%p'), 
    description="Retrieve and return the current date and/or time. \
        Input should be an empty string.",
)

    tools = [search_tool, datetimetool]

    TEMPLATE = """Answer the following questions as best you can, You have access to the following tool:
    {tools}
    Always use the following format:
    Chat History: {chat_history}
    Question: the input question you must answer
    Thought: you should always think if you need to search the web to answer this question.
    Action: if you know the answer and you don't think it's necessary to search the web, you can directly answer the question (skip to Final Answer).
    Otherwise, if web search is necessary, you can use this tool to search the web [{tool_names}].
    Action Input: the input to the action (i.e., the search query you will use).
    Observation: the result of the action (i.e., the information retrieved from the web).
    ... (this Thought/Action/Action Input/Observation sequence can repeat multiple times)
    Final Thought: I now know the final answer.
    Final Answer: the complete final answer to the original input question.
    Begin!

    Chat History:{chat_history}
    Question: {input}
    Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(
        template=TEMPLATE,
        partial_variables={"date": datetime.now(), 'chat_history': chat_history},
    )
    # Create the agent
    agent = create_react_agent(llm=LLM, tools=tools, prompt=prompt)

    # Agent executor
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True,
    )
    return agent_executor

