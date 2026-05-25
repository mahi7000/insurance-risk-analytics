import pandas as pd
import numpy as np

def load_insurance_data(file_path: str) -> pd.DataFrame:
    """Loads the ACIS insurance dataset and runs basic type casting."""
    df = pd.read_csv(file_path)
    
    # Cast dates safely
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        
    # Calculate target KPI metrics explicitly
    df['LossRatio'] = np.where(df['TotalPremium'] > 0, df['TotalClaims'] / df['TotalPremium'], 0)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    
    return df