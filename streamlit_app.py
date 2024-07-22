import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
import os

# Set up environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# ------------------- Create Sidebar Chat ----------------------

# Increase the default width of the main area by 50%
st.set_page_config(layout="wide")


with st.sidebar:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.subheader('**How to Use:**')
    st.write('''

    1. 📄 Upload Wastage and Maintenance files.
    2. Wait for the files to upload.
    3. Ask Questions! 📊

    ''')
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

    uploaded_file_w = st.file_uploader("**Upload Wastage File (.csv)**", type=("csv"))
    uploaded_file_m = st.file_uploader("**Upload Maintenance File (.csv)**", type=("csv"))

    if uploaded_file_w and uploaded_file_m:
        dfe = pd.read_csv(uploaded_file_w)
        dfr = pd.read_csv(uploaded_file_m)
    
        # Connect to the SQLite database
        conn = sqlite3.connect('Data.db')
        dfe.to_sql('Wastage_Data', conn, index=False, if_exists='replace')
        dfr.to_sql('Maintenance_Data', conn, index=False, if_exists='replace')
        
        # Commit and close the connection
        conn.commit()
        conn.close()
        st.success("Files have been successfully uploaded.")



st.title('🤖 Data to Insights')
st.markdown("#### Unlock Actionable Insights from Your Machine Data")
st.markdown("")
st.markdown("")


# Step 1: Define example queries
examples = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
    {"input": "Find all albums for the artist 'AC/DC'.", "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');"},
    {"input": "List all tracks in the 'Rock' genre.", "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');"},
]

# Step 2: Create a FewShotPromptTemplate
example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a SQL expert. Given a user input, generate the appropriate SQL query.\nHere are some examples:",
    suffix="User input: {input}\nSQL query: {agent_scratchpad}\n",
    input_variables=["input", "agent_scratchpad"]
)

# Step 3: Create a full prompt template with MessagesPlaceholder
full_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=few_shot_prompt),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Initialize the LLM and create the SQL agent
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
db = SQLDatabase.from_uri("sqlite:///Data.db")
agent = create_sql_agent(llm, db=db, prompt=full_prompt, agent_type="openai-tools", verbose=False)


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
        
        # Construct a dictionary with required inputs
        inputs = {"input": user_query, "agent_scratchpad": ""}
        
        # Call the agent's run method with the inputs dictionary
        response = agent.run(callbacks=[st_cb], **inputs)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)






# # def get_fewshot_agent_chain(): 
    
#     # llm & db setup
# llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
#     # db = SQLDatabase.from_uri("sqlite:///Data.db")

#     # # create few shot prompts, their embeddings and store in Chromadb
#     # embeddings = OpenAIEmbeddings()
    
#     # # create example selector which chooses k= examples to include in the agent's prompt
#     # example_selector = SemanticSimilarityExampleSelector.from_examples(
#     #     few_shots_ag,
#     #     embeddings,
#     #     Chroma,
#     #     k=3,
#     #     input_keys=["input"],
#     # )


#     # # Now we can create our FewShotPromptTemplate, which takes our example selector, an example prompt for formatting each example, and a string prefix and suffix to put before and after our formatted examples:
#     # from langchain_core.prompts import (
#     #     ChatPromptTemplate,
#     #     FewShotPromptTemplate,
#     #     MessagesPlaceholder,
#     #     PromptTemplate,
#     #     SystemMessagePromptTemplate,
#     # )

#     # system_prefix = """You are an agent designed to interact with a SQL database.
#     # Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
#     # Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
#     # You can order the results by a relevant column to return the most interesting examples in the database.
#     # Never query for all the columns from a specific table, only ask for the relevant columns given the question.
#     # You have access to tools for interacting with the database.
#     # Only use the given tools. Only use the information returned by the tools to construct your final answer.
#     # You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

#     # DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

#     # If the question does not seem related to the database, just return "I don't know" as the answer.

#     # Here are some examples of user inputs and their corresponding SQL queries:"""

#     # few_shot_prompt = FewShotPromptTemplate(
#     #     example_selector=example_selector,
#     #     example_prompt=PromptTemplate.from_template(
#     #         "User input: {input}\nSQL query: {query}"
#     #     ),
#     #     input_variables=["input", "dialect", "top_k"],
#     #     prefix=system_prefix,
#     #     suffix="",
#     # )

#     # # our full prompt should be a chat prompt with a human message template and an agent_scratchpad MessagesPlaceholder.
#     # full_prompt = ChatPromptTemplate.from_messages(
#     #     [
#     #         SystemMessagePromptTemplate(prompt=few_shot_prompt),
#     #         ("human", "{input}"),
#     #         MessagesPlaceholder("agent_scratchpad"),
#     #     ]
#     # )

#     # agent_executor = create_sql_agent(llm, db=db, prompt=full_prompt, agent_type="openai-tools", verbose=True)
#     # return agent_executor




