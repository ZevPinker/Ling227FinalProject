Run make_csv.py to load data from web
Run data_reader.py to format the csv into the training validation and test text files 
Run feature2id.py to create the feature dict which is stored in data/features.csv, and then use that to create the training, validation, and test sets
Run model_trainer.py to load data sets, train the model and then evaluate (lists of tuples which contain feature tensors and label)