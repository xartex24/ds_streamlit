import pandas as pd

def format_dataframe(df):
    """
    Format numeric columns in European number format (1.000,25)
    while preserving specific columns like years
    """
    # Create a copy of the dataframe to avoid modifying the original
    formatted_df = df.copy()
    
    numeric_cols = formatted_df.select_dtypes(include=['float64', 'int64']).columns
    
    
    exclude_cols = ['year', 'Year', 'annee', 'Annee']
    
    # Format numeric columns
    for col in numeric_cols:
        if col in exclude_cols:
            continue
        
        if formatted_df[col].dtype == 'int64':
            formatted_df[col] = formatted_df[col].apply(lambda x: f'{x:,.0f}'.replace(',', '.') if pd.notnull(x) else x)
        
        elif formatted_df[col].dtype == 'float64':
            formatted_df[col] = formatted_df[col].apply(lambda x: 
                f'{x:,.2f}'.replace(',', 'TEMP').replace('.', ',').replace('TEMP', '.') 
                if pd.notnull(x) else x
            )
    
    return formatted_df

# Define paths to datasets
def get_data_paths():
    return {
        "raw_data": "C:/Users/julia/OneDrive/Documents/04_DataScientest/01_Data Analyst/Bikes_Paris/01_Data/01_External/comptage-velo-donnees-compteurs.csv",
        "clean_data": "C:/Users/julia/OneDrive/Documents/04_DataScientest/01_Data Analyst/Bikes_Paris/01_Data/02_Cleaned/bike_df_cleaned.csv"
    }