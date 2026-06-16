import pandas as pd

def clean_data(df):
    df = df.drop_duplicates()

    drop_cols = [
        "Customer Email",
        "Product Image",
        "Customer Password",
        "Customer Street",
        "Customer Fname",
        "Customer Lname",
        "Order Zipcode",
        "Customer Zipcode"
    ]

    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].astype(str).fillna("Unknown")

    return df