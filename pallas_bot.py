import os
import sys
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION
# ==========================================
# REPLACE with your actual OpenAI API Key
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# REPLACE 'password' with your MySQL Root Password
# We connect to the 'pallas_poc' database you just created
db_uri = "mysql+mysqlconnector://root:root@localhost:3306/pallas_db"

# ==========================================
# 2. SCHEMA MAPPING (The "Brain") - UPDATED FOR ALL 8 TABLES
# ==========================================
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

# Update the database connection to include ALL table keys
db = SQLDatabase.from_uri(
    db_uri,
    include_tables=list(custom_table_info.keys()), # Now includes all 8 tables
    custom_table_info=custom_table_info
)

# ==========================================
# 3. FEW-SHOT EXAMPLES (The "Training")
# ==========================================
examples = [
    # 1. Sales & Items per Hour for Specific Stores (Q2 2025)
    {
        "input": "Sales, Number of item sold per store per day per HOUR per store for store AH2301, AH1080, AH5823 for Q2-2025",
        "sql_cmd": """
        SELECT 
            f.OMSCH AS Store_Name, 
            d.DATUM AS Date, 
            tv.UREN2 AS Hour_of_Day, 
            SUM(t.KLANTENBIJDRAGE) AS Total_Sales, 
            SUM(t.CE_AANTAL) AS Total_Items
        FROM L4TRANSACTIEREGEL t
        JOIN FILIAAL f ON t.FILA_ID = f.ID
        JOIN DAG d ON t.DAGE_ID = d.ID
        JOIN TIJDVAK tv ON t.TIJK_ID = tv.ID
        WHERE 
            d.JAAR_ID = 2025 
            AND d.KWAR_ID = 2 
            AND f.OMSCH IN ('AH2301', 'AH1080', 'AH5823')
        GROUP BY f.OMSCH, d.DATUM, tv.UREN2;
        """
    },

    # 2. Transaction Count per Half-Hour for AH2GO Stores (Period 7 2025)
    {
        "input": "Number of transactions per half hour per day per store AH2GO stores for Period 7 2025",
        "sql_cmd": """
        SELECT 
            f.OMSCH AS Store_Name, 
            d.DATUM AS Date, 
            tv.UREN_MINUTEN AS Time_Bucket, 
            COUNT(DISTINCT t.TRAN_ID_TR) AS Transaction_Count
        FROM L4TRANSACTIEREGEL t
        JOIN FILIAAL f ON t.FILA_ID = f.ID
        JOIN DAG d ON t.DAGE_ID = d.ID
        JOIN TIJDVAK tv ON t.TIJK_ID = tv.ID
        WHERE 
            d.JAAR_ID = 2025 
            AND d.PERI_ID = 7 
            AND f.OMSCH LIKE '%AH2GO%'
        GROUP BY f.OMSCH, d.DATUM, tv.UREN_MINUTEN;
        """
    },

    # 3. Sales per WAG (Product Group) for Franchise Stores (Week 45 2025)
    {
        "input": "Sales, Number of item sold per store per day per WAG for AH2GO Franchise stores for week 45-2025",
        "sql_cmd": """
        SELECT 
            f.OMSCH AS Store_Name, 
            d.DATUM AS Date, 
            w.WAG_OMSCH AS Product_Group, 
            SUM(t.KLANTENBIJDRAGE) AS Total_Sales, 
            SUM(t.CE_AANTAL) AS Total_Items
        FROM L4TRANSACTIEREGEL t
        JOIN FILIAAL f ON t.FILA_ID = f.ID
        JOIN DAG d ON t.DAGE_ID = d.ID
        JOIN ARTIKEL a ON t.ARTI_ID = a.ID
        JOIN WAG w ON a.WAGF_ID = w.ID
        WHERE 
            d.JAAR_ID = 2025 
            AND d.WEEK_ID = 45 
            AND f.OMSCH LIKE '%AH2GO%' 
            AND f.FRANCHISE_IND = 'J'
        GROUP BY f.OMSCH, d.DATUM, w.WAG_OMSCH;
        """
    }
]

# ==========================================
# 4. BUILD THE CHAIN
# ==========================================
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
    examples=examples,
    example_prompt=example_prompt,
    prefix=system_prefix,
    suffix="User Input: {input}\nSQL Query:",
    # ERROR FIX: Added "top_k" to this list, and {table_info} is now in the prefix above
    input_variables=["input", "table_info", "top_k"], 
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=api_key,
    temperature=0)

chain = create_sql_query_chain(llm, db, prompt=prompt)

# ==========================================
# 5. INTERACTIVE LOOP
# ==========================================
def run_pallas_bot():
    print("\n🤖 Pallas SQL Bot is Ready! (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nAsk Pallas: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
            
        try:
            print("Thinking...")
            response = chain.invoke({"question": user_input})
            
            # Clean up formatting
            cleaned_sql = response.replace("```sql", "").replace("```", "").strip()
            
            print("\n>> Generated MySQL Query:")
            print(cleaned_sql)
            print("-" * 50)
            
        except Exception as e:
            print(f"Error generating SQL: {e}")

if __name__ == "__main__":
    run_pallas_bot()
