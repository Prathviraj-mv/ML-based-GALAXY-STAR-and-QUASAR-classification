import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('data/SDSS_DR18.csv')
df = df.dropna(subset=['u', 'g', 'r', 'i', 'z', 'redshift', 'class'])

label_encoder = LabelEncoder()
df['object_class_encoded'] = label_encoder.fit_transform(df['class'])

X = df[['u', 'g', 'r', 'i', 'z', 'redshift']]
y = df['object_class_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print(" Model and label encoder saved!")
