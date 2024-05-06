

import pandas as pd
import csv
import torch
import torch.nn as nn

import argparse

import random
import numpy as np

import logging

import twokenize
from string import ascii_lowercase, ascii_uppercase, digits



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



def tokenize_post(post:str): 
    stripped = preprocess_text(post)
    tokens = twokenize.tokenize(stripped)

    return tokens


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



def read_dataset_from_npy(filename):
    dataset = []

    features, labels = np.load(filename, allow_pickle=True)
    for feature, label in zip(features, labels):
        dataset.append((torch.tensor(feature, dtype=torch.float), torch.tensor(label, dtype=torch.long)))

    return dataset


#####################################################################################
# DEFINE NEURAL NETWORK ARCHITECTURES
#####################################################################################

class LogisticClassifier(nn.Module):

    def __init__(self, n_input, n_output):
        super(LogisticClassifier, self).__init__()

        self.layer1 = nn.Linear(n_input, n_output)

    def forward(self, inp):

        # Note: To get our predictions, we should apply a softmax
        # to the vector "outp". However, it turns out that, in
        # PyTorch, the loss function we will be using (which is
        # CrossEntropyLoss) incorporates a softmax. Thus, we do not
        # need to include a softmax inside the model architecture.
        outp = self.layer1(inp)

        return outp


# Currently, the definition of this class has been copied from the
# definition of LogisticClassifier above. Modify it so that it instantiates
# a multi-layer perceptron (MLP) with two layers.
#
# Specifically, to go from inp to outp, you should first apply a linear
# layer to the input, whose output should be a vector of size hidden_size.
# Then, you should apply a ReLU to that vector.
# Then, you should apply dropout with a proportion of dropout_p
# Finally, you should apply another linear layer that maps the output of the
# dropout layer to a vector of size n_output.
# You should not include a Softmax or Sigmoid at the end of "forward", for the
# reasons noted in the comment within "forward".
#
# It may be helpful to examine the SentimentClassifier class from the PyTorch
# tutorial posted on Canvas. For implementing dropout, it may be helpful to
# check out the PyTorch documentation for Dropout at the following link:
# https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
class MLP2(nn.Module):

    def __init__(self, n_input, n_output, hidden_size=100, dropout_p=0.0):
        super(MLP2, self).__init__()

        # Define the first linear layer
        self.layer1 = nn.Linear(n_input, hidden_size)
        # Define the second linear layer
        self.layer2 = nn.Linear(hidden_size, n_output)
        # Define dropout layer
        self.dropout = nn.Dropout(p=dropout_p)
        # Define ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, inp):
        # Apply the first linear layer
        outp = self.layer1(inp)
        # Apply ReLU activation function
        outp = self.relu(outp)
        # Apply dropout
        outp = self.dropout(outp)
        # Apply the second linear layer
        outp = self.layer2(outp)

        return outp

# Modify this class so that it instantiates a multilayer perceptron (MLP) with
# three layers. This should be similar to MLP2 above, but with one extra layer.
#
# Specifically, the forward pass should include these components, in order:
# - First apply a linear layer to the input that maps it to a vector of
#   size hidden_size.
# - Then apply ReLU to that vector
# - Then apply dropout, with a proportion of dropout_p
# - Then apply another linear layer, mapping this vector (which is of size
#   hidden_size) to another vector of size hidden_size.
# - Then apply a ReLU to that vector
# - Then apply dropout, with a proportion of dropout_p
# - And finally apply another linear layer that maps this vector (which has
#   a size of hidden_size) to the final output, which should be a vector of
#   size n_output


class MLP3(nn.Module):

    def __init__(self, n_input, n_output, hidden_size=100, dropout_p=0.0):
        super(MLP3, self).__init__()

        # Define the first linear layer
        self.layer1 = nn.Linear(n_input, hidden_size)
        # Define the second linear layer
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        # Define the third linear layer
        self.layer3 = nn.Linear(hidden_size, n_output)
        # Define dropout layers
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        # Define ReLU activation functions
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, inp):
        # Apply the first linear layer
        outp = self.layer1(inp)
        # Apply ReLU activation function
        outp = self.relu1(outp)
        # Apply dropout
        outp = self.dropout1(outp)
        # Apply the second linear layer
        outp = self.layer2(outp)
        # Apply ReLU activation function
        outp = self.relu2(outp)
        # Apply dropout
        outp = self.dropout2(outp)
        # Apply the third linear layer
        outp = self.layer3(outp)

        return outp


#####################################################################################
# FUNCTION FOR EVALUATING ON A DATASET
# Can be applied to the validation set or the test set
#####################################################################################

def compute_evaluation_loss(model, eval_set, loss_function):
    total_loss = 0
    total_correct = 0

    count_examples = 0
    for example in eval_set:
        pred = model(example[0]).unsqueeze(0)
        correct = example[1]

        loss = loss_function(pred, correct)
        total_loss += loss
        count_examples += 1

        _, pred_label = torch.topk(pred, 1)
        pred_label = pred_label.item()
        if pred_label == correct.item():
            total_correct += 1

    evaluation_loss = total_loss / count_examples
    evaluation_accuracy = total_correct / count_examples

    return evaluation_loss.item(), evaluation_accuracy

#####################################################################################
# FUNCTION FOR TRAINING
#####################################################################################


def random_search(num_iterations):
    # Define ranges for hyperparameters
    lr_range = [0.0001, 0.0012, 0.0011, 0.001, 0.01]
    dropout_range = [0.0, 0.1, 0.2]
    hidden_size_range = [50, 100, 150]
    batch_size_range = [1, 10, 32]
    eval_every_range = [2000, 5000, 10000]

    best_accuracy = 0
    best_hyperparameters = {}

    for _ in range(num_iterations):
        # Randomly sample hyperparameters
        lr = random.choice(lr_range)
        dropout = random.choice(dropout_range)
        hidden_size = random.choice(hidden_size_range)
        batch_size = random.choice(batch_size_range)
        eval_every = random.choice(eval_every_range)

        # Train the model with the current hyperparameters
        model = MLP3(input_feature_count, label_count,
                     hidden_size=hidden_size, dropout_p=dropout)
        train(model, training_set, validation_set, test_set, eval_on_test=args.eval_on_test, lr=lr,
              n_epochs=args.n_epochs, batch_size=batch_size, eval_every=eval_every, model_name="mlp")

        # Evaluate the model
        _, accuracy = compute_evaluation_loss(
            model, validation_set, nn.CrossEntropyLoss())

        # Update best hyperparameters if accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparameters = {
                "lr": lr,
                "dropout": dropout,
                "hidden_size": hidden_size,
                "batch_size": batch_size,
                "eval_every": eval_every
            }

    return best_hyperparameters


def train(model, training_set, validation_set, test_set, eval_on_test=False, lr=0.001, n_epochs=1, batch_size=1, eval_every=1000, model_name="mlp"):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    training_examples_seen = 0
    best_loss = 1000000

    for epoch in range(n_epochs):

        for example_input, example_output in training_set:

            prediction = model(example_input)
            label = example_output

            # The "unsqueeze" is necessary to get the right
            # sizes for the tensors here
            loss = loss_function(prediction.unsqueeze(0), label)
            loss.backward()

            if training_examples_seen % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            if training_examples_seen % eval_every == 0:
                validation_loss, validation_accuracy = compute_evaluation_loss(
                    model, validation_set, loss_function)
                print(epoch, training_examples_seen, "Loss:",
                      validation_loss, "Accuracy:", validation_accuracy)

                if validation_loss < best_loss:
                    best_loss = validation_loss
                    torch.save(model.state_dict(), model_name + ".weights")

            training_examples_seen += 1

    # Once we are done training, we evaluate
    model.load_state_dict(torch.load(model_name + ".weights"))
    validation_loss, validation_accuracy = compute_evaluation_loss(
        model, validation_set, loss_function)
    print("\nFINAL VALIDATION METRICS:")
    print("Validation loss:", validation_loss,
          "Validation accuracy:", validation_accuracy)

    if eval_on_test:
        test_loss, test_accuracy = compute_evaluation_loss(
            model, test_set, loss_function)
        print("\nTEST METRICS:")
        print("Test loss:", test_loss, "Test accuracy:", test_accuracy)


if __name__ == "__main__":
    logging.info("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", help="training file",
                        type=str, default="data/train.txt")
    parser.add_argument("--validation", "-v", help="validation file",
                        type=str, default="data/validate.txt")
    parser.add_argument("--test", "-e", help="test file",
                        type=str, default="data/test.txt")
    parser.add_argument(
        "--eval_on_test", help="evaluate on the test set", action='store_true')
    parser.add_argument("--lr", help="learning rate",
                        type=float, default=0.001)
    parser.add_argument(
        "--dropout", help="dropout percentage", type=float, default=0.0)
    parser.add_argument(
        "--hidden_size", help="hidden size for the MLP", type=int, default=100)
    parser.add_argument(
        "--n_epochs", help="number of times we loop over the training set", type=int, default=10)
    parser.add_argument(
        "--batch_size", help="batch size", type=int, default=1)
    parser.add_argument(
        "--eval_every", help="this is the number n where we evaluate on the validation set after every n examples", type=int, default=5000)
    parser.add_argument(
        "--glove", help="Use concatenated GloVe embeddings as the input features", action='store_true')
    parser.add_argument(
        "--model", help="type of model (mlp2, mlp3, or logistic)", type=str, default="logistic")
    parser.add_argument(
        "--random_seed", help="random seed",
                        type=int, default=42)
    parser.add_argument(
        "--num_iterations", help="number of random search iterations", type=int, default=10)

    args = parser.parse_args()
    

    # Setting Hyper-Parameters manually here
    # args.hidden_size = 128
    # args.dropout = 0.1
    # args.lr = .003
    # args.n_epochs = 6
    # args.batch_size = 10
    # args.eval_every = 5000

    print("Hyperparameters:")
    print(f"- Hidden Size: {args.hidden_size}")
    print(f"- Dropout: {args.dropout}")
    print(f"- Learning Rate (lr): {args.lr}")
    print(f"- Number of Epochs: {args.n_epochs}")
    print(f"- Batch Size: {args.batch_size}")
    print(f"- Evaluation Frequency: {args.eval_every}")

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    label_count = 4

    training_set= read_dataset_from_npy('data/training_set.csv.npy')
    validation_set = read_dataset_from_npy('data/validation_set.csv.npy')
    test_set = read_dataset_from_npy('data/test_set.csv.npy')

    input_feature_count = len(training_set[0][0])
    if args.model == "logistic":
        model = LogisticClassifier(input_feature_count, label_count)
    elif args.model == "mlp2":
        model = MLP2(input_feature_count, label_count,
                     hidden_size=args.hidden_size, dropout_p=args.dropout)
    elif args.model == "mlp3":
        model = MLP3(input_feature_count, label_count,
                     hidden_size=args.hidden_size, dropout_p=args.dropout)
    # train(model, training_set, validation_set, test_set, eval_on_test=args.eval_on_test, lr=args.lr,
    #       n_epochs=args.n_epochs, batch_size=args.batch_size, eval_every=args.eval_every, model_name="mlp")
    best_hyperparameters = random_search(args.num_iterations)
    print("Best Hyperparameters:", best_hyperparameters)

    logging.info("finish")



