def calculate_kpis(df):

    total_orders = len(df)

    delayed_orders = df["Late_delivery_risk"].sum()

    delay_rate = (
        delayed_orders
        /
        total_orders
    ) * 100

    total_sales = df["Sales"].sum()

    total_profit = df["Order Profit Per Order"].sum()

    return {
        "Total Orders": total_orders,
        "Delayed Orders": delayed_orders,
        "Delay Rate": round(delay_rate, 2),
        "Total Sales": round(total_sales, 2),
        "Total Profit": round(total_profit, 2)
    }