import pandas as pd

csv_files = ['Documents/yellow_tripdata_2015-01.csv', 'Documents/yellow_tripdata_2016-01.csv', 'Documents/yellow_tripdata_2016-02.csv', 'Documents/yellow_tripdata_2016-03.csv']

dataframes = [pd.read_csv(file) for file in csv_files]
df = pd.concat(dataframes, ignore_index=True)

df.to_csv('yellow_tripdata.csv', index=False)
