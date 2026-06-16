import pandas as pd

def load_data(path):
    try:
        df = pd.read_csv(path, encoding="latin1")
        print("✅ Data Loaded Successfully")
        return df

    except Exception as e:
        print("❌ Error:", e)
        return None