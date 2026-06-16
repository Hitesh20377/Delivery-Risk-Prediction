import numpy as np

def create_features(df):

    df["Delay_Gap"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]

    df["Shipping_Pressure_Index"] = df["Order Item Quantity"] / (
        df["Days for shipment (scheduled)"] + 1
    )

    df["Order_Complexity_Score"] = df["Order Item Quantity"] * df["Product Price"]

    df["Profit_Sensitivity"] = df["Order Profit Per Order"] / (df["Sales"] + 1)

    # inf / -inf ko NaN banao
    df = df.replace([np.inf, -np.inf], np.nan)

    # bache hue NaN fill karo
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(0)

    return df