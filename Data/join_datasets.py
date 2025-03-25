import pandas as pd

csv_root_folder = '../Documents/'
csv_files = ['yellow_tripdata_2015-01.csv', 'yellow_tripdata_2016-01.csv', 'yellow_tripdata_2016-02.csv', 'yellow_tripdata_2016-03.csv']

dataframes = [pd.read_csv(csv_root_folder + file) for file in csv_files]
df = pd.concat(dataframes, ignore_index=True)

csv_output_folder = '../Data/'
df.to_csv(csv_output_folder + 'yellow_tripdata.csv', index=False)
