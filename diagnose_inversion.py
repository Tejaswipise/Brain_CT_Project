import pandas as pd
import numpy as np

df = pd.read_csv('data/features_slices.csv')

print("high_hu_fraction by PATIENT (not class):")
print(df.groupby(['patient_id','class_name','scanner_model'])
        ['high_hu_fraction'].mean()
        .reset_index()
        .sort_values('high_hu_fraction', ascending=False)
        .to_string(index=False))

print("\nhigh_hu_fraction by SCANNER within class:")
print(df.groupby(['class_name','scanner_model'])
        ['high_hu_fraction'].mean()
        .reset_index()
        .to_string(index=False))

print("\nRevolution patients only (no scanner confound):")
rev = df[df['scanner_model'].str.contains('Revolution')]
print(rev.groupby(['class_name'])
         ['high_hu_fraction'].mean()
         .reset_index()
         .to_string(index=False))

print("\nhu_p90 by patient:")
print(df.groupby(['patient_id','class_name','scanner_model'])
        ['hu_p90'].mean()
        .reset_index()
        .sort_values('hu_p90', ascending=False)
        .to_string(index=False))