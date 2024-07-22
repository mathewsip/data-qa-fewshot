import streamlit as st
from langchain_community.utilities import SQLDatabase
import pandas as pd
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from few_shots_for_agent import few_shots_ag
import os
import sqlite3



os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true" 
db = SQLDatabase.from_uri("sqlite:///Data.db")



def get_fewshot_agent_chain(): 
    
    # llm & db setup
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    db = SQLDatabase.from_uri("sqlite:///Data.db")

    # create few shot prompts, their embeddings and store in Chromadb
    embeddings = OpenAIEmbeddings()
    
    # create example selector which chooses k= examples to include in the agent's prompt
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        few_shots_ag,
        embeddings,
        Chroma,
        k=3,
        input_keys=["input"],
    )


    # Now we can create our FewShotPromptTemplate, which takes our example selector, an example prompt for formatting each example, and a string prefix and suffix to put before and after our formatted examples:
    from langchain_core.prompts import (
        ChatPromptTemplate,
        FewShotPromptTemplate,
        MessagesPlaceholder,
        PromptTemplate,
        SystemMessagePromptTemplate,
    )

    system_prefix = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I don't know" as the answer.

    Here are some examples of user inputs and their corresponding SQL queries:"""

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k"],
        prefix=system_prefix,
        suffix="",
    )

    # our full prompt should be a chat prompt with a human message template and an agent_scratchpad MessagesPlaceholder.
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent_executor = create_sql_agent(llm, db=db, prompt=full_prompt, agent_type="openai-tools", verbose=True)
    return agent_executor







st.title("🔎 Natural language question-answering with databases")
"""
In this app we're using text-to-SQL AI to convert questions asked in natural language into the SQL queries needed to extract data from a DB to answer them. The app uses multiple technologies including GPT3.5 to generate the SQL queries, Langchain to manage the question-answering process, and OpenAI Embeddings and Chroma to build a database of typical questions and their correct answers to provide examples to the AI, i.e. implementing few-shot prompting to improve accuracy.
"""


tab1, tab2 = st.tabs([":page_facing_up: Agent", ":robot_face: Golden SQL"])

with tab1:

    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask me anything!")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            agent = get_fewshot_agent_chain()
            response = agent.run(user_query, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)


with tab2:

    st.markdown("")
    st.markdown("**Q&A examples used in this implementation:**")
    df = pd.DataFrame(few_shots_ag)
    st.table(df)


