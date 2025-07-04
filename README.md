# ML-based-GALAXY-STAR-and-QUASAR-classification
~~~


 ░▒▓███████▓▒░   ░▒▓██████▓▒░    ░▒▓██████▓▒░    ░▒▓██████▓▒░  
░▒▓█▓▒░         ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░ 
░▒▓█▓▒░         ░▒▓█▓▒          ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒       
 ░▒▓██████▓▒░   ░▒▓█▓▒▒▓███▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒        
       ░▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒        
       ░▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒  ▒▓█▓▒░  ░▒▓█▓▒   ▒▓█▓▒░ 
░▒▓███████▓▒░    ░▒▓██████▓▒░   ░▒▓██████▓▒░     ░▒▓██████▓▒░  
                                 ░▒▓█▓▒░                   
    STAR          GALAXY         QUASAR           CLASSIFICATION

~~~
# TRAIN RANDOM FOREST MODEL
~~~
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('data/SDSS_DR18.csv')
df = df.dropna(subset=['u', 'g', 'r', 'i', 'z', 'redshift', 'class'])

# Feature
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


label_encoder = LabelEncoder()
df['object_class_encoded'] = label_encoder.fit_transform(df['class'])

features = [
    'u-g', 'g-r', 'r-i', 'i-z', 'u-r', 'g-i', 'r-z',
    'g/r', 'r/i', '(u-g)/(g-r)', 'log_redshift'
]

X = df[features]
y = df['object_class_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GridSearch
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}

print("Tuning Random Forest parameters, please wait...")
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)

# Evaluation
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# F Importance
importances = best_model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar([features[i] for i in sorted_indices], importances[sorted_indices])
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

joblib.dump(best_model, 'modelold.pkl')
joblib.dump(label_encoder, 'label_encoderold.pkl')
print("\nall good to go @_@")
~~~
# TEST RANDOM FOREST TRAINED MODEL

<p>
 <img src="star.png" width=50%>
</p>
<p>
 <img src="qsr.png" width=50%>
</p>
<p>
 <img src="galaxy.png" width=50%>
</p>

![Figure_1](https://github.com/user-attachments/assets/469164fb-b8fe-4ac5-b82c-4b354e1d8a97)

![Screenshot 2025-07-04 174943](https://github.com/user-attachments/assets/7bffac9c-8120-4b6b-90ff-d114109bf3eb)






~~~
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
~~~
