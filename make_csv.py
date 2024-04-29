from datasets import load_dataset

dataset = load_dataset("OsamaBsher/AITA-Reddit-Dataset")

# Access the MemoryMappedTable
table = dataset["train"].data
# convert it to a pandas df
table = table.to_pandas()

table.to_csv('data/data.csv', index=False)  # Set index=False to exclude row indices


