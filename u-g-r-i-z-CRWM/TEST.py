import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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

dataread = pd.read_csv(input_file)
dataread = dataread.dropna(subset=['u', 'g', 'r', 'i', 'z', 'redshift'])

# Feature engineering
dataread['u-g'] = dataread['u'] - dataread['g']
dataread['g-r'] = dataread['g'] - dataread['r']
dataread['r-i'] = dataread['r'] - dataread['i']
dataread['i-z'] = dataread['i'] - dataread['z']
dataread['u-r'] = dataread['u'] - dataread['r']
dataread['g-i'] = dataread['g'] - dataread['i']
dataread['r-z'] = dataread['r'] - dataread['z']
dataread['g/r'] = dataread['g'] / dataread['r']
dataread['r/i'] = dataread['r'] / dataread['i']
dataread['(u-g)/(g-r)'] = dataread['u-g'] / dataread['g-r']
dataread['log_redshift'] = np.log10(dataread['redshift'] + 1)

features = [
    'u-g', 'g-r', 'r-i', 'i-z', 'u-r', 'g-i', 'r-z',
    'g/r', 'r/i', '(u-g)/(g-r)', 'log_redshift'
]


predictions_encoded = model.predict(dataread[features])
predictions = label_encoder.inverse_transform(predictions_encoded)


stars_count = 0
galaxies_count = 0
quasars_count = 0

star_types = []
galaxy_types = []
quasar_types = []

print("\nResults ---\n")
for idx, pred in enumerate(predictions, start=1):
    obj = dataread.iloc[idx - 1]
    classification_detail = ""

    if pred == 'STAR':
        stars_count += 1
        if obj['g-r'] < 0.0:
            classification_detail = "O/B STAR"
        elif obj['g-r'] < 0.3:
            classification_detail = "A STAR"
        elif obj['g-r'] < 0.6:
            classification_detail = "F/G STAR"
        elif obj['g-r'] < 1.0:
            classification_detail = "K STAR"
        else:
            classification_detail = "M STAR"
        star_types.append(classification_detail)

    elif pred == 'GALAXY':
        galaxies_count += 1
        if obj['g-r'] > 0.7:
            classification_detail = "Elliptical GALAXY"
        elif obj['g-r'] > 0.3:
            classification_detail = "Spiral GALAXY"
        else:
            classification_detail = "Irregular GALAXY"
        galaxy_types.append(classification_detail)

    elif pred in ['QSO', 'QUASAR']:
        quasars_count += 1
        z = obj['redshift']
        if z < 1.0:
            classification_detail = "Low-z QUASAR"
        elif z < 2.5:
            classification_detail = "Mid-z QUASAR"
        else:
            classification_detail = "High-z QUASAR"
        quasar_types.append(classification_detail)

    print(f"{idx}. {pred} -> {classification_detail}")

# Summary counts
print(f"\nStars   : {stars_count}")
print(f"Galaxies: {galaxies_count}")
print(f"Quasars : {quasars_count}")

print("\nDetailed Classification Summary:")
print("Stars Breakdown:", dict(Counter(star_types)))
print("Galaxies Breakdown:", dict(Counter(galaxy_types)))
print("Quasars Breakdown:", dict(Counter(quasar_types)))

# Bar
labels = ['Stars', 'Galaxies', 'Quasars']
counts = [stars_count, galaxies_count, quasars_count]
plt.figure(figsize=(8, 6))
plt.bar(labels, counts, color=['blue', 'green', 'red'])
plt.xlabel('Object Type')
plt.ylabel('Count')
plt.title('Distribution of Stars, Galaxies, Quasars')
plt.tight_layout()
plt.show()

# Pie
if star_types:
    star_counts = dict(Counter(star_types))
    plt.figure()
    plt.pie(star_counts.values(), labels=star_counts.keys(), autopct='%1.1f%%')
    plt.title("Star Types Breakdown")

if galaxy_types:
    galaxy_counts = dict(Counter(galaxy_types))
    plt.figure()
    plt.pie(galaxy_counts.values(), labels=galaxy_counts.keys(), autopct='%1.1f%%')
    plt.title("Galaxy Types Breakdown")

if quasar_types:
    quasar_counts = dict(Counter(quasar_types))
    plt.figure()
    plt.pie(quasar_counts.values(), labels=quasar_counts.keys(), autopct='%1.1f%%')
    plt.title("Quasar Types Breakdown")





data = pd.read_csv(input_file)

class_column = 'class'

if class_column not in data.columns:
    print(f"The column '{class_column}'")
else:
    class_counts = data[class_column].value_counts()



    stars = class_counts.get('STAR', 0)
    galaxies = class_counts.get('GALAXY', 0)
    quasars = class_counts.get('QSO', 0) + class_counts.get('QUASAR', 0)

print(f"\nactual no. of stars: {stars}")
print(f"actual no. of Galaxies: {galaxies}")
print(f"actual no. of Quasars: {quasars}")

print(f"\nStars   : {stars_count}")
print(f"Galaxies: {galaxies_count}")
print(f"Quasars : {quasars_count}")

def compute_percentage_error(predicted, actual, label):
    error = abs(predicted - actual) / actual * 100
    print(f"{label} count error: {error:.2f}% (Predicted: {predicted}, Actual: {actual})")

compute_percentage_error(stars_count, stars, "Stars")
compute_percentage_error(galaxies_count, galaxies, "Galaxies")
compute_percentage_error(quasars_count, quasars, "Quasars")

plt.show()


#STAR
# O/B: very blue (low u-g, low g-r)
#
# A/F: moderate blue
#
# G/K: yellow-orange
#
# M: red (high g-r, r-i)
#QSR
# Low-z: z < 1
#
# Mid-z: 1 ≤ z < 2.5
#
# High-z: z ≥ 2.5
