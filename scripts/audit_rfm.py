import pandas as pd
import numpy as np

def audit():
    df = pd.read_parquet('data/gold/gold_customer_rfm.parquet')
    cols = ['recency_days', 'frequency', 'monetary_gross', 'monetary_net', 'return_amount', 'return_orders_count']
    print('=== DESCRIPTIVE STATISTICS ===')
    desc = df[cols].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]).T
    print(desc[['count', 'mean', 'std', 'min', '1%', '5%', '50%', '95%', '99%', '99.9%', 'max']].to_string())

    print('\n=== SKEWNESS & KURTOSIS ===')
    for col in cols:
        print(f'{col:20}: skew = {df[col].skew():8.2f}, kurtosis = {df[col].kurtosis():8.2f}')

    print('\n=== NEGATIVE & ZERO NET MONETARY CUSTOMERS ===')
    zero_net = df[df['monetary_net'] == 0]
    neg_net = df[df['monetary_net'] < 0]
    pos_net = df[df['monetary_net'] > 0]
    print(f'monetary_net == 0: {len(zero_net)} customers')
    print(f'monetary_net < 0 : {len(neg_net)} customers (min={neg_net["monetary_net"].min():.2f}, max={neg_net["monetary_net"].max():.2f}, sum={neg_net["monetary_net"].sum():.2f})')
    print(f'monetary_net > 0 : {len(pos_net)} customers (min={pos_net["monetary_net"].min():.2f}, max={pos_net["monetary_net"].max():.2f}, sum={pos_net["monetary_net"].sum():.2f})')

    print('\n=== ZERO FREQUENCY CUSTOMERS (No valid purchases) ===')
    zero_freq = df[df['frequency'] == 0]
    print(f'frequency == 0   : {len(zero_freq)} customers')
    if len(zero_freq) > 0:
        print(zero_freq[['customer_id', 'frequency', 'monetary_gross', 'monetary_net', 'return_amount', 'return_orders_count']].head(10).to_string())

    print('\n=== EXTREME SPENDING OUTLIERS (Top 10 Monetary Net) ===')
    print(df.nlargest(10, 'monetary_net')[['customer_id', 'recency_days', 'frequency', 'monetary_gross', 'monetary_net', 'return_amount', 'customer_segment']].to_string())

    print('\n=== EXTREME NEGATIVE OUTLIERS (Bottom 5 Monetary Net) ===')
    print(df.nsmallest(5, 'monetary_net')[['customer_id', 'recency_days', 'frequency', 'monetary_gross', 'monetary_net', 'return_amount', 'customer_segment']].to_string())

    print('\n=== RULE-BASED CUSTOMER SEGMENTS ===')
    print(df['customer_segment'].value_counts().to_string())

if __name__ == '__main__':
    audit()
