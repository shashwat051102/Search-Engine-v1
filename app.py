import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
# DuckDuckgosearchrun helps you to search anything on the internet
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
# StreamlitCallbackHandler it is used to handle all the tools and agents within the webapp
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

# Arxiv and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name = "Search")


st.title("Search engine v1")

# sidebar for settings
# st.sidebar.title("Settings")
api_keys = os.getenv("GROQ_API_KEY")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant","content":"Hi, I am an chat bot who also search the web. How can I help you"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Whenever you are going to write any prompt in the chatbot it will be stored in the messages
# and the chatbot will respond to that prompt
if prompt:=st.chat_input(placeholder="What is machine learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    
    
    llm = ChatGroq(groq_api_key = api_keys, model_name = "Llama-3.3-70b-Versatile",streaming=True)
    tools = [search,arxiv,wiki]
    
    search_agents = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agents.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)
        
