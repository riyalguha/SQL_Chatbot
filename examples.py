# examples.py

# This list holds the training data for the Pallas Bot
pallas_examples = [
    # 1. Sales & Items per Hour
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

    # 2. Transaction Count per Half-Hour
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

    # 3. Sales per WAG (Product Group)
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
