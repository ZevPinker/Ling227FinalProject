import numpy as np
import csv
from string import ascii_lowercase, ascii_uppercase, digits
import twokenize
import logging
import torch
# Configure logging
logging.basicConfig(filename='trainer.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def preprocess_text(text):
    stripped = ""
    for char in text:
        if char == '&':
            stripped += 'and'
        elif char == ' ':
            stripped += ' '
        elif char in ascii_lowercase:
            stripped += char
        elif char in ascii_uppercase:
            stripped += char.lower()
        elif char in digits:
            stripped += char
    return stripped


def tokenize_post(post: str):
    stripped = preprocess_text(post)
    tokens = twokenize.tokenize(stripped)

    return tokens

# Takes in 3 files, then finds all features used in those files. Creates a dictionary that has features
# as keys with numerical IDs as values. That is, the first feature that
# shows up in the files will get an ID of 0, the second will get an ID of 1,
# etc. This will be important for converting our features into tensors that PyTorch can use.


def feature_counter(train_filename, validation_filename, test_filename):
    feature2id = {}  # A dictionary converting each feature to a numerical ID
    feature_count = 0  # The number of features that we have
    features_list = []  # List to store features and their IDs

    # Now loop over our files and find all features  Each feature will be added to our feature2id
    # dictionary
    filenames = [train_filename, validation_filename, test_filename]
    line_no = 0
    for filename in filenames:
        logging.info("reading: " + filename)
        with open(filename, "r") as f:
            for line in f:
                line_no += 1
                print("reading line " + str(line_no) + "/" + str(270710*.15))
                # Extracting text portion after the label tag
                text = line[5:].strip()
                # Tokenize the text
                tokens = tokenize_post(text)
                # Update feature2id dictionary and features_list
                for token in tokens:
                    if token not in feature2id:
                        feature2id[token] = feature_count
                        features_list.append((token, feature_count))
                        feature_count += 1

    # Write features to a CSV file
    with open('data/features.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Feature', 'ID'])
        writer.writerows(features_list)

    return feature2id, feature_count


def read_features_csv(csv_filename):
    feature2id = {}  # A dictionary converting each feature to a numerical ID

    # Read features from CSV file
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            feature = row['Feature']
            feature_id = int(row['ID'])  # Convert ID to integer
            feature2id[feature] = feature_id

    return feature2id


# Creates a dataset from a file listing features.
# The dataset is a list of (input, output) pairs.
# The input should be a PyTorch vector whose length is the number of features
# that we have. This tensor should include a 1 at the vector positions corresponding
# to the features that appear in this example, with zeroes everywhere else. For example,
# if this example included features 1 and 7 (but no others), the vector would look like:
#      torch.tensor([0,1,0,0,0,0,0,1,0,0,0, ... ,0,0])
# The output should be a PyTorch tensor containing a single integer corresponding
# to the correct label for this example. Thus, if the example belongs to class 0, the
# output should be torch.tensor([0]); if it belongs to class 1, the output
# should be torch.tensor([1]).


def do_label2id(label: str):
    if label == "yta":
        return torch.tensor([0])
    elif label == "nta":
        return torch.tensor([1])
    elif label == "nah":
        return torch.tensor([2])
    elif label == "esh":
        return torch.tensor([3])

# returns a binary tensor which tells us which features are in the post.
# This is one-hot encoding


def do_text2feature(post: str, feature2id):
    tokens = tokenize_post(post)

    # Initialize a tensor with zeros
    feature_tensor = torch.zeros(len(feature2id), dtype=torch.float)

    # Loop through tokens and update the corresponding feature tensor values
    for token in tokens:
        if token in feature2id:
            # Setting the value to 1.0 if the feature is present
            feature_tensor[feature2id[token]] = 1.0

    return feature_tensor


# A data set is a list of  (feature_tensor[], label_tensor[])
#
def create_dataset_from_file(filename, feature2id):
    dataset = []

    with open(filename, 'r') as f:
        line_no = 0
        for line in f:
            line_no += 1
            print("featurizing " + str(line_no) + "/" + str(int(40606/3)))
            # Splitting the line into label and text
            label, text = line.strip().split(' ', 1)
            # Removing '<' and '>' from label
            label = label[1:-1]

            dataset.append(
                (do_text2feature(text, feature2id), (do_label2id(label))))

    return dataset


def save_dataset_as_npy(dataset, filename):
    features = [data[0].numpy() for data in dataset]
    labels = [data[1].numpy() for data in dataset]
    np.save(filename, (features, labels))




if __name__ == '__main__':
    feature_counter('data/train.txt', 'data/validate.txt', 'data/test.txt')

    feature2id = read_features_csv('data/features.csv')
    feature_count = len(feature2id)
    label_count = 4

    training_set = create_dataset_from_file(
        'data/train.txt', feature2id=feature2id)
    validation_set = create_dataset_from_file(
        'data/validate.txt', feature2id=feature2id)
    test_set = create_dataset_from_file(
        'data/test.txt', feature2id=feature2id)

    save_dataset_as_npy(training_set, 'data/training_set')
    save_dataset_as_npy(validation_set, 'data/validation_set')
    save_dataset_as_npy(test_set, 'data/test_set')



