import pandas as pd

dataset_root_folder = '../Data/'
df = pd.read_csv(dataset_root_folder + 'yellow_tripdata.csv', parse_dates=['tpep_pickup_datetime'])

# Delete columns
df = df.drop([
    'VendorID', 
    'tpep_dropoff_datetime', 
    'passenger_count',       
    'trip_distance',         
    'RateCodeID',            
    'store_and_fwd_flag',    
    'dropoff_longitude',     
    'dropoff_latitude',      
    'payment_type',          
    'fare_amount',           
    'extra',                 
    'mta_tax',               
    'improvement_surcharge', 
    'tip_amount',            
    'tolls_amount',          
    'total_amount'           
], axis=1)

# Delete rows with null values 
df = df.dropna()

# Delete duplicates
df = df.drop_duplicates()

# Delete negative latitude values because NY is around 40
df = df[df["pickup_latitude"] > 0]

# Delete positive latitude values because NY is around -73
df = df[df["pickup_longitude"] < 0]

# Segmentation of the pickup datetime 
df['year'] = df['tpep_pickup_datetime'].dt.year
df['month'] = df['tpep_pickup_datetime'].dt.month
df['day'] = df['tpep_pickup_datetime'].dt.day
df['hour'] = df['tpep_pickup_datetime'].dt.hour

df.drop(columns=['tpep_pickup_datetime'], inplace=True)

# Filter pickup hours from 6am to 12pm
df = df[df['hour'] >= 6]

# Approximate coordinates values
df['pickup_longitude'] = df['pickup_longitude'].round(3)
df['pickup_latitude'] = df['pickup_latitude'].round(3)

# Group by the data by the date and the coordinates   
df_grouped = df.groupby(['year', 'month', 'day', 'hour', 'pickup_longitude', 'pickup_latitude']).size().reset_index(name='demand')

# Save the new dataset
df_grouped.to_csv(dataset_root_folder + 'taxis_dataset.csv', index=False)