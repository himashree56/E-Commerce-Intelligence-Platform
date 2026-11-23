import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

def profile_clusters():
    df = pd.read_parquet('data/gold/gold_customer_rfm.parquet')
    
    # Feature Set A
    X_A = pd.DataFrame({
        'recency': np.log1p(df['recency_days']),
        'frequency': np.log1p(df['frequency']),
        'monetary': signed_log1p(df['monetary_net']),
    })
    scaler_A = StandardScaler()
    X_scaled_A = scaler_A.fit_transform(X_A)
    
    for k in [4, 5]:
        print(f"\n{'='*35} SET A: k = {k} {'='*35}")
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_scaled_A)
        df_k = df.copy()
        df_k['cluster'] = labels
        
        prof = df_k.groupby('cluster').agg(
            count=('customer_id', 'count'),
            recency_med=('recency_days', 'median'),
            recency_mean=('recency_days', 'mean'),
            freq_med=('frequency', 'median'),
            freq_mean=('frequency', 'mean'),
            monetary_med=('monetary_net', 'median'),
            monetary_mean=('monetary_net', 'mean'),
            monetary_sum=('monetary_net', 'sum'),
            return_amt_med=('return_amount', 'median'),
            return_amt_mean=('return_amount', 'mean'),
            pct_neg_net=('monetary_net', lambda s: (s <= 0).mean() * 100),
        ).reset_index()
        prof['pop_pct'] = (prof['count'] / len(df_k) * 100).round(2)
        prof['rev_pct'] = (prof['monetary_sum'] / df_k['monetary_net'].sum() * 100).round(2)
        print(prof.to_string())
        
        print("\nRule-based segments per cluster:")
        ct = pd.crosstab(df_k['cluster'], df_k['customer_segment'])
        print(ct.to_string())

if __name__ == '__main__':
    profile_clusters()
