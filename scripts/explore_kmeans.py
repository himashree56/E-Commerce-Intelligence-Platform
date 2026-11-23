import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def signed_log1p(x):
    """Symmetric log1p transform preserving sign for negative and positive values."""
    return np.sign(x) * np.log1p(np.abs(x))

def run_experiment():
    df = pd.read_parquet('data/gold/gold_customer_rfm.parquet')
    
    # Feature Set A: Classic RFM (Recency, Frequency, Monetary Net with signed log1p)
    # Feature Set B: RFM + Returns (Recency, Frequency, Monetary Net, Return Amount)
    # Feature Set C: RFM Gross + Returns (Recency, Frequency, Monetary Gross, Return Amount)
    
    X_A = pd.DataFrame({
        'recency': np.log1p(df['recency_days']),
        'frequency': np.log1p(df['frequency']),
        'monetary': signed_log1p(df['monetary_net']),
    })
    
    X_B = pd.DataFrame({
        'recency': np.log1p(df['recency_days']),
        'frequency': np.log1p(df['frequency']),
        'monetary': signed_log1p(df['monetary_net']),
        'returns': np.log1p(df['return_amount']),
    })
    
    X_C = pd.DataFrame({
        'recency': np.log1p(df['recency_days']),
        'frequency': np.log1p(df['frequency']),
        'monetary_gross': np.log1p(df['monetary_gross']),
        'returns': np.log1p(df['return_amount']),
    })
    
    experiments = [
        ('Set A (R, F, M_net) - StandardScaler', X_A, StandardScaler()),
        ('Set A (R, F, M_net) - RobustScaler', X_A, RobustScaler()),
        ('Set B (R, F, M_net, Returns) - StandardScaler', X_B, StandardScaler()),
        ('Set B (R, F, M_net, Returns) - RobustScaler', X_B, RobustScaler()),
        ('Set C (R, F, M_gross, Returns) - StandardScaler', X_C, StandardScaler()),
        ('Set C (R, F, M_gross, Returns) - RobustScaler', X_C, RobustScaler()),
    ]
    
    print('=' * 85)
    print('        K-MEANS CLUSTERING HYPERPARAMETER & FEATURE EXPLORATION')
    print('=' * 85)
    
    for name, X_raw, scaler in experiments:
        X_scaled = scaler.fit_transform(X_raw)
        print(f'\n--- {name} ---')
        print(f'{"k":3} | {"Inertia":10} | {"Silhouette":10} | {"Davies-Bouldin":14} | {"Calinski-Harabasz":17} | Cluster Sizes')
        print('-' * 85)
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
            sizes = pd.Series(labels).value_counts().sort_index().to_dict()
            size_str = ', '.join([f'{cnt}' for c_id, cnt in sizes.items()])
            print(f'{k:3} | {km.inertia_:10.1f} | {sil:10.4f} | {db:14.4f} | {ch:17.1f} | [{size_str}]')

if __name__ == '__main__':
    run_experiment()
