from langchain_community.tools import DuckDuckGoSearchRun

search_tool=DuckDuckGoSearchRun()

result=search_tool.run("What is the Today News")
print(result)