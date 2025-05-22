import joblib
import pandas as pd
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
print("STARS ,GALAXY ,QUASARS DETECTION USING PHOTOMETRIC DATA")

model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

input_file = input("Enter the path to the CSV file: ")

data = pd.read_csv(input_file)
print(data)
total_rows = len(data)

print(f"Total number of rows: {total_rows}")

stars_count = 0
galaxies_count = 0
quasars_count = 0

predictions = []

for index, row in data.iterrows():
    u = float(row['u'])
    g = float(row['g'])
    r = float(row['r'])
    i = float(row['i'])
    z = float(row['z'])
    redshift = float(row['redshift'])

    new_data = pd.DataFrame([[u, g, r, i, z, redshift]], columns=['u', 'g', 'r', 'i', 'z', 'redshift'])
    predicted_encoded = model.predict(new_data)[0]
    predicted_class = label_encoder.inverse_transform([predicted_encoded])[0]

    print(f"Row {index+1} → Predicted: {predicted_class}")

    predictions.append(predicted_class)

    if predicted_class.upper() == 'STAR':
        stars_count += 1
    elif predicted_class.upper() == 'GALAXY':
        galaxies_count += 1
    elif predicted_class.upper() in ['QSO', 'QUASAR']:
        quasars_count += 1

print(f"\nNumber of stars: {stars_count}")
print(f"Number of Galaxies: {galaxies_count}")
print(f"Number of quasars: {quasars_count}")



# Plot the results
labels = ['Quasars', 'Galaxies', 'stars']
counts = [stars_count, galaxies_count, quasars_count]

plt.figure(figsize=(8, 6))
plt.bar(labels, counts, color=['blue', 'green', 'red'])
plt.xlabel('Object Type')
plt.ylabel('Count')
plt.title('Distribution of Stars, Galaxies, and Quasars')
plt.show()
