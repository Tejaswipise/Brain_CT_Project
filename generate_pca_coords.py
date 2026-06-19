import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

df = pd.read_csv('data/data/features_patients.csv')
feat_cols = [c for c in df.columns if c.endswith('_mean') or c.endswith('_max')]

X_real = df[feat_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_real)

pca = PCA(n_components=2, random_state=42)
coords_real = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame({
    'patient_id':    df['patient_id'].values,
    'class_name':    df['class_name'].values,
    'scanner_model': df['scanner_model'].values,
    'class_label':   df['class_label'].values,
    'scanner_label': df['scanner_label'].values,
    'is_synthetic':  False,
    'pc1':           coords_real[:, 0],
    'pc2':           coords_real[:, 1],
})

# Add synthetic patients if that file exists
synth_path = Path('data/data/synthetic_features.csv')
if synth_path.exists():
    df_synth = pd.read_csv(synth_path)
    X_synth_scaled = scaler.transform(df_synth[feat_cols].values)
    coords_synth = pca.transform(X_synth_scaled)
    df_synth_pca = pd.DataFrame({
        'patient_id':    df_synth['patient_id'].values,
        'class_name':    df_synth['class_name'].values,
        'scanner_model': df_synth['scanner_model'].values,
        'class_label':   df_synth['class_label'].values,
        'scanner_label': df_synth['scanner_label'].values,
        'is_synthetic':  True,
        'pc1':           coords_synth[:, 0],
        'pc2':           coords_synth[:, 1],
    })
    df_pca = pd.concat([df_pca, df_synth_pca], ignore_index=True)
    print(f"Added {len(df_synth_pca)} synthetic patients")

df_pca.to_csv('data/data/pca_coordinates.csv', index=False)
print(f"Saved: data/data/pca_coordinates.csv")
print(df_pca[['patient_id', 'class_name', 'scanner_model', 'pc1', 'pc2']].to_string(index=False))