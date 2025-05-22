import pandas as pd

input_file = 'data/SDSS_DR18.csv'
data = pd.read_csv(input_file)

class_column = 'class'

if class_column not in data.columns:
    print(f"The column '{class_column}' is not found in the CSV file.")
else:
    class_counts = data[class_column].value_counts()
    stars_count = class_counts.get('STAR', 0)
    galaxies_count = class_counts.get('GALAXY', 0)
    quasars_count = class_counts.get('QSO', 0) + class_counts.get('QUASAR', 0)

    print(f"\nNumber of stars: {stars_count}")
    print(f"Number of Galaxies: {galaxies_count}")
    print(f"Number of Quasars: {quasars_count}")
