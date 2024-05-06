import pandas as pd
import os

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('data/data.csv')

# Print the head of the DataFrame


# Function to write data to file
def write_to_file(file_path, data):
    with open(file_path, 'w') as f:
        for index, row in data.iterrows():
            f.write(f"<{row['verdict']}> {row['text']}\n")


# Splitting the data into train, validate, and test sets
# These values can be changed to include more or less data, my machine only seems to be able to handle 15% of the 270k posts
train_data = df[:int(0.01*len(df))] 
validate_data = df[int(0.01*len(df)):int(0.02*len(df))]
test_data = df[int(0.02*len(df)):int(0.03*len(df))]

# Write data to files
write_to_file(os.path.join('data', 'train.txt'), train_data)
write_to_file(os.path.join('data', 'validate.txt'), validate_data)
write_to_file(os.path.join('data', 'test.txt'), test_data)
