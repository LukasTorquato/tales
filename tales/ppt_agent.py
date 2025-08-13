# MCP imports
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import RunnableConfig
from tales.prompts import powerpoint_prompt
from tales.config import server_params, llm, RUNNABLE_CONFIG

async def ppt_agent(msgs):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print("Initializing session...")
            await session.initialize()
            print("Loading tools...")
            tools = await load_mcp_tools(session)
            # print(tools)
            print("Creating agent...")
            # llm.bind_tools(tools)
            agent = create_react_agent(
                model=llm, tools=tools, prompt=powerpoint_prompt.content
            )

            print("Invoking agent...")
            query = ""
            for m in msgs:
                query += m.content + "\n"

            response = await agent.ainvoke({"messages": query},config=RUNNABLE_CONFIG)

            print("Printing response...")
            for message in response["messages"]:
                message.pretty_print()
    