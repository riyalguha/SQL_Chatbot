import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv

# Import your examples from the separate file
try:
    from examples import pallas_examples
except ImportError:
    st.error("Could not find 'examples.py'. Please ensure it exists in the same directory.")
    pallas_examples = []

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Pallas SQL Bot",
    page_icon="🤖",
    layout="centered"
)

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("This POC connects to the local **Pallas MySQL** database.")
    
    # Reload button to clear cache if you change code
    if st.button("Reset Chat Session"):
        st.session_state.messages = []
        st.rerun()

# --- SETUP FUNCTION (Cached for Performance) ---
@st.cache_resource
def setup_chain():
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        st.error("GEMINI_API_KEY not found in .env file.")
        return None

    # Database Connection
    db_uri = "mysql+mysqlconnector://root:root@localhost:3306/pallas_db"
    
    # Schema Definition
    custom_table_info = {
    "l4transactieregel": """
        SALES_FACTS_LINE_ITEMS: Individual items sold in a transaction.
        - KLANTENBIJDRAGE: Sales Amount for this specific item.
        - CE_AANTAL: Quantity of this specific item.
        - ARTI_ID: Link to ARTIKEL.
        - FILA_ID: Link to FILIAAL.
        - TRAN_ID_TR: Link to parent transaction in L4TRANSACTIE.
        """,
    "l4transactie": """
        SALES_HEADERS: The total transaction/receipt summary (Basket level).
        - TRAN_ID_TR: Unique Transaction ID.
        - KLANTENBIJDRAGE: Total amount of the whole receipt/basket.
        - CE_AANTAL: Total number of items in the basket.
        - FILA_ID: Store ID.
        - DAGE_ID: Date ID.
        """,
    "artikel": """
        PRODUCTS: Item details.
        - OMSCH: Product Name.
        - ID: Product ID.
        - WAGF_ID: Link to WAG (Product Group).
        - ASSF_ID: Link to ASSORTIMENTSGROEP.
        """,
    "filiaal": """
        STORES: Branch details.
        - OMSCH: Store Name/Code.
        - PLAATSNAAM: City.
        - FRANCHISE_IND: 'J' = Franchise.
        """,
    "dag": """
        DATE_DIMENSION: Calendar.
        - JAAR_ID: Year (2024, 2025).
        - WEEK_ID: Week (1-52).
        - PERI_ID: Period (1-13).
        - MAAN_ID: Month (1-12).
        """,
    "tijdvak": """
        TIME_DIMENSION: Time of day.
        - UREN2: Hour (0-23).
        """,
    "wag": """
        PRODUCT_GROUPS: High-level category (Warengroep).
        - WAG_OMSCH: Group Name.
        """,
    "assortimentsgroep": """
        ASSORTMENT_GROUPS: A middle-layer product hierarchy.
        - ASSORTIMENTSGROEP_OMSCH: Assortment Name.
        - ID: Links to ARTIKEL.ASSF_ID.
        """
}

    try:
        db = SQLDatabase.from_uri(
            db_uri,
            include_tables=list(custom_table_info.keys()),
            custom_table_info=custom_table_info
        )
    except Exception as e:
        st.error(f"Database Connection Failed: {e}")
        return None

    # Prompt Setup
    system_prefix = """
You are a MySQL expert for the 'Pallas' retail system.
Your goal is to generate valid MySQL queries based on the user's question.

CRITICAL RULES:
1. Return ONLY the SQL code. Do not wrap it in markdown (no ```sql).
2. Use the Schema Map provided to understand column names (e.g., OMSCH is Name, KLANTENBIJDRAGE is Revenue).
3. Always JOIN tables correctly using the IDs provided in the schema descriptions.
4. Unless the user specifies otherwise, limit the results to {top_k}.

Here is the database schema you must use:
{table_info}

Below are some examples of how to solve questions:
"""

    example_prompt = PromptTemplate.from_template("User Input: {input}\nSQL Query: {sql_cmd}")

    prompt = FewShotPromptTemplate(
        examples=pallas_examples,
        example_prompt=example_prompt,
        prefix=system_prefix,
        suffix="User Input: {input}\nSQL Query:",
        input_variables=["input", "table_info", "top_k"], 
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key, temperature=0)
    
    return create_sql_query_chain(llm, db, prompt=prompt)

# --- INITIALIZE CHAIN ---
chain = setup_chain()

# --- CHAT INTERFACE ---
st.title("🤖 Pallas Text-to-SQL")
st.caption("Ask questions about Sales, Stores, and Products in natural language.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am Pallas. Ask me a question like: \n*Sales per store for week 45 2025*"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "SELECT" in message["content"]:
            st.code(message["content"], language="sql")
        else:
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # 1. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        if chain:
            with st.spinner("Generating SQL Query..."):
                try:
                    response = chain.invoke({"question": prompt})
                    cleaned_sql = response.replace("```sql", "").replace("```", "").strip()
                    
                    st.markdown("Here is the SQL query for your request:")
                    st.code(cleaned_sql, language="sql")
                    
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": cleaned_sql})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("Chain not initialized. Check Database connection.")
