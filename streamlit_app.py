import streamlit as st
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
import os
import sqlite3
from langchain.tools import Tool
from datetime import datetime

# Set up environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Ensure the database is empty at the start of the session
db_path = 'Data.db'
if os.path.exists(db_path):
    os.remove(db_path)

# Connect to the SQLite database (this will create a new, empty database)
conn = sqlite3.connect(db_path)

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

    1. 📄 Upload one or two relevant files in CSV format (e.g. maintenance data, customer data, sales data, etc.)
    2. Example test case: upload wastage data on the left and maintenance data on the right.
    3. Wait for the files to upload.
    4. Ask Questions! 📊

    ''')
    st.markdown("")
    st.markdown("")
    st.markdown("")


st.title('🤖 Data to Insights')
st.markdown("#### Unlock Actionable Insights from Your Process Data")
st.markdown("")
st.markdown("")

col1, col2 = st.columns(2)

with col1:
    uploaded_file_1 = st.file_uploader("**Upload File 1 (e.g. wastage, customer, sales or finance data) (.csv)**", type=("csv"))

with col2:
    uploaded_file_2 = st.file_uploader("**Upload File 2 (e.g. maintenance, customer, sales or finance data) (.csv)**", type=("csv"))

# Connect to the SQLite database
conn = sqlite3.connect('Data.db')

if uploaded_file_1:
    # Read the uploaded file
    dfe = pd.read_csv(uploaded_file_1)
    
    # Save dataframe to SQL table
    dfe.to_sql('File 1', conn, index=False, if_exists='replace')
    
    st.success("File 1 successfully uploaded and data ready for analysis.")

if uploaded_file_2:
    # Read the uploaded file
    dfr = pd.read_csv(uploaded_file_2)
    
    # Save dataframe to SQL table
    dfr.to_sql('File 2', conn, index=False, if_exists='replace')
    
    st.success("File 2 successfully uploaded and data ready for analysis.")

# Commit and close the connection
conn.commit()
conn.close()


st.markdown("")
st.markdown("")
st.markdown("")


# Step 1: Define example queries
examples = [
    {"input": "What is the amount of wastage for tea blends?", "query": """SELECT "Copy of Comp MatlGrp Desc" AS ComponentMaterialGroup, SUM("Var2Prf Amt") AS TotalWastage FROM Wastage_Data WHERE "Copy of Comp MatlGrp Desc" = 'Tea Blends' GROUP BY "Copy of Comp MatlGrp Desc";"""},
    {"input": "What's the reasons for plastic bags wastage on L1?", "query": "SELECT Level2Reason, COUNT(*) AS ReasonCount \nFROM Maintenance_Data \nWHERE Line = 'L01 - C24' \nGROUP BY Level2Reason;"},
    {"input": "What was the top contributor to wastage this month?", "query": """SELECT "Copy of Comp MatlGrp Desc" AS ComponentMaterialGroup, SUM("Var2Prf Amt") AS TotalWastage \nFROM Wastage_Data \nGROUP BY "Copy of Comp MatlGrp Desc";"""},
]

# Step 2: Create a FewShotPromptTemplate
example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    # prefix="You are a SQL expert. Given a user input, generate the appropriate SQL query.\nHere are some examples:",
    prefix="""You are an assitant for process engineers. You are an agent designed to interact with a SQL database.
    Given an input question about data, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer. 
    You can order the results by a relevant column to return the most interesting examples in the database. 
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.,
    You have access to tools for interacting with the database.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I don't know" as the answer. \nHere are some examples:""",
    suffix="User input: {input}\nSQL query: {agent_scratchpad}\n",
    input_variables=["input", "agent_scratchpad"]
)

# Step 3: Create a full prompt template with MessagesPlaceholder
full_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=few_shot_prompt),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


# OLD PROMPT
    # When a user asks about a material or item, they are referring to a unique entity from the column 'Copy of Comp MatlGrp Desc' column in the 'Wastage_Data' table with only these values possible: ['Tea Blends', 'ZWIP Default', 'Thermal Transfer Lbl', 'Corrugated & Display', 'Web', 'Misc Pkg Materials', 'ASSO BRAND DELTA MFG', '0', 'Cartons', 'Tea Tags', 'PS Labels', 'Poly Laminations', 'ZFIN DEFAULT', 'Plastic Bags']  
    # When asked about 'downtime', 'reasons' or 'maintenance' query the 'Maintenance_Data' table.
    # 'Reasons' for downtime and maintenance are provided as Level 2 Reasons in the Maintenance_Data table in the column 'Level2Reason'.
    # When asked about Lines or, for example, "L1", the lines you can query are only: ['L01 - C24', 'L02 - C24', 'L03 - C24', 'L03A - C24E', 'L04 - C21', 'L05  - C21', 'L19 - T2 Prima', 'L21 - Twinkle', 'L22 - Twinkle Rental', 'L23 - Twinkle 2', 'L24 - Twinkle 3', 'L35 - Fuso Combo 1', 'L36 - Fuso Combo 2']



# # Custom function to get the current date
# def get_current_date():
#     return datetime.now().strftime("%Y-%m-%d")

# # Create a tool from the custom function
# date_tool = Tool(
#     name="get_current_date",
#     func=get_current_date,
#     description="Get the current date"
# )

# # Simple test function
# def simple_test_tool():
#     return "Test tool response for Ian"

# # Create a tool from the simple test function
# test_tool_Ian = Tool(
#     name="simple_test_tool",
#     func=simple_test_tool,
#     description="Returns a test response"
# )

# Initialize the LLM and create the SQL agent
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
db = SQLDatabase.from_uri("sqlite:///Data.db")
# agent = create_sql_agent(llm, db=db, prompt=full_prompt, tools=[date_tool, test_tool_Ian], agent_type="openai-tools", verbose=True)
agent = create_sql_agent(llm, db=db, prompt=full_prompt, agent_type="openai-tools", verbose=True)


if "messages" not in st.session_state or st.sidebar.button("New Conversation"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, how can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything about your data!")

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

