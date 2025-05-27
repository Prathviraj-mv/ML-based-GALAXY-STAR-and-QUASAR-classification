# TEST DATA
~~~
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('''

 ░▒▓███████▓▒░   ░▒▓██████▓▒░    ░▒▓██████▓▒░    ░▒▓██████▓▒░  
░▒▓█▓▒░         ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░ 
░▒▓█▓▒░         ░▒▓█▓▒          ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒       
 ░▒▓██████▓▒░   ░▒▓█▓▒▒▓███▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒        
       ░▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒        
       ░▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒   ▒▓█▓▒░ 
░▒▓███████▓▒░    ░▒▓██████▓▒░   ░▒▓██████▓▒░     ░▒▓██████▓▒░  
                                 ░▒▓█▓▒░                   

''')
print("STARS, GALAXY, QUASARS DETECTION USING PHOTOMETRIC DATA\n")

model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

input_file = "data/SDSS_DR18.csv"
# input_file = "SDSS17.csv"
print("\n started...")

df = pd.read_csv(input_file)
df = df.dropna(subset=['u', 'g', 'r', 'i', 'z', 'redshift'])

# Feature Engineering
df['u-g'] = df['u'] - df['g']
df['g-r'] = df['g'] - df['r']
df['r-i'] = df['r'] - df['i']
df['i-z'] = df['i'] - df['z']
df['u-r'] = df['u'] - df['r']
df['g-i'] = df['g'] - df['i']
df['r-z'] = df['r'] - df['z']
df['g/r'] = df['g'] / df['r']
df['r/i'] = df['r'] / df['i']
df['(u-g)/(g-r)'] = df['u-g'] / df['g-r']
df['log_redshift'] = np.log10(df['redshift'] + 1)

features = [
    'u-g', 'g-r', 'r-i', 'i-z', 'u-r', 'g-i', 'r-z',
    'g/r', 'r/i', '(u-g)/(g-r)', 'log_redshift'
]

# Predict
predictions_encoded = model.predict(df[features])
predictions = label_encoder.inverse_transform(predictions_encoded)

# Counters
stars_count = 0
galaxies_count = 0
quasars_count = 0

print("\n--- Prediction Results ---\n")
for idx, pred in enumerate(predictions, start=1):
    print(f"{idx}. Predicted Class: {pred}")

    if pred == 'STAR':
        stars_count += 1
    elif pred == 'GALAXY':
        galaxies_count += 1
    elif pred in ['QSO', 'QUASAR']:
        quasars_count += 1

# Summary
print("\n--- Summary ---")
print(f"Total Stars   : {stars_count}")
print(f"Total Galaxies: {galaxies_count}")
print(f"Total Quasars : {quasars_count}")

# Bar Chart
labels = ['Stars', 'Galaxies', 'Quasars']
counts = [stars_count, galaxies_count, quasars_count]

plt.figure(figsize=(8, 6))
plt.bar(labels, counts, color=['blue', 'green', 'red'])
plt.xlabel('Object Type')
plt.ylabel('Count')
plt.title('Distribution of Stars, Galaxies, and Quasars')
plt.tight_layout()
plt.show()
~~~
