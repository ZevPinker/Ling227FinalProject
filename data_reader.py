import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('data/data.csv')

# Print the head of the DataFrame
print(df.head())

print(df["text"][0])