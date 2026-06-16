import numpy as np
import pandas as pd

MODEL_DROP_COLS = [
    "Late_delivery_risk",
    "Delivery Status",
    "Days for shipping (real)",
    "Delay_Gap",

    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Email",
    "Product Image",

    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",
    "Order Item Cardprod Id",
    "Product Card Id",
    "Product Category Id",
    "Category Id",
    "Department Id",

    "Order Zipcode",
    "Order City",
    "Order State",
    "Customer City",
    "Customer State",
    "Product Name",
    "Product Description",

    "order date (DateOrders)",
    "shipping date (DateOrders)",
]


def get_model_features(df):
    X = df.drop(
        columns=[col for col in MODEL_DROP_COLS if col in df.columns],
        errors="ignore"
    )

    X = X.replace({pd.NA: np.nan})

    return X