import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import twokenize
from string import ascii_lowercase, ascii_uppercase, digits
import argparse
import random
import logging

# Configure logging
logging.basicConfig(filename='app.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

#####################################################################################
# READ IN TRAINING DATA AND PROCESS INTO A WORD2VEC MODEL
#####################################################################################

# Function to preprocess text


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


# Load data
df = pd.read_csv('data/data.csv')
posts = df['text'][0]

# Preprocess text and tokenize
stripped_text = preprocess_text(posts)
corpus = twokenize.tokenize(stripped_text)
logging.info(corpus)
# Function to generate word pairs from tokenized corpus


def generate_word_pairs(corpus, window_size):
    word_pairs = []
    for i, target_word in enumerate(corpus):
        context = corpus[max(0, i - window_size):i] + \
            corpus[i + 1:i + window_size + 1]
        for context_word in context:
            word_pairs.append((target_word, context_word))
    return word_pairs

# Function to preprocess text and generate word pairs

# Hyperparameters for Word2Vec
window_size = 2
vector_size = 100
min_count = 1
epochs = 10
lr = 0.01  # Learning rate for Word2Vec training

# Initialize the Word2Vec model


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, target_word_idx):
        embedded = self.embeddings(target_word_idx)
        predicted_context = self.linear(embedded)
        return predicted_context


def train_word2vec(corpus, vocab_size, embedding_size, context_size, epochs, lr):
    model = Word2Vec(vocab_size, embedding_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for target_word, context_words in corpus:
            optimizer.zero_grad()
            target_word_idx = torch.LongTensor([target_word])
            context_word_indices = torch.LongTensor(context_words)
            predicted_context = model(target_word_idx)
            loss = criterion(predicted_context, context_word_indices)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss}')
    return model


# Generate word pairs
word_pairs = generate_word_pairs(corpus, window_size)

# Index words
word2idx = {word: idx for idx, (word, _) in enumerate(set(word_pairs))}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx)
indexed_corpus = [(word2idx[target], word2idx[context])
                  for target, context in word_pairs]


# Takes in a pair of files, then finds all features and class
# labels used in those files. Creates a dictionary that has features
# as keys with numerical IDs as values. That is, the first feature that
# shows up in the files will get an ID of 0, the second will get an ID of 1,
# etc. This will be important for converting our features into tensors that PyTorch can use.
# Similarly, also creates a dictionary that has labels as keys with
# numerical IDs as values (i.e., the first label that shows up gets
# an ID of 0, the second gets an ID of 1, etc. - since we are dealing with
# a binary classification task, there should only be 2 unique labels).

# Configure logging
logging.basicConfig(level=logging.INFO)


def feature_counter(train_filename, validation_filename, test_filename):
    feature2id = {}  # A dictionary converting each feature to a numerical ID
    feature_count = 0  # The number of features that we have

    label2id = {}  # A dictionary converting each class label to a numerical ID
    label_count = 0  # The number of labels that we have

    # Now loop over our files and find all features and class labels
    # that are used there. Each feature will be added to our feature2id
    # dictionary, and each label will be added to our label2id dictionary
    filenames = [train_filename, validation_filename, test_filename]
    for filename in filenames:
        logging.debug(f"Processing file: {filename}")
        with open(filename, "r") as fi:
            for line in fi:
                features = line.strip().split(",")[:-1]

                for feature in features:
                    if feature not in feature2id:
                        feature2id[feature] = feature_count
                        feature_count += 1
                        logging.info(
                            f"Added feature: {feature} with ID: {feature_count - 1}")
                    else:
                        logging.debug(f"Feature already exists: {feature}")

                label = line.strip().split(",")[-1]
                if label not in label2id:
                    label2id[label] = label_count
                    label_count += 1
                    logging.info(
                        f"Added label: {label} with ID: {label_count - 1}")
                else:
                    logging.debug(f"Label already exists: {label}")

    return feature2id, feature_count, label2id, label_count


def create_dataset_from_tokens(tokenized_corpus, window_size, feature_count, feature2id, label_count=None, label2id=None):
    # Generate word pairs from tokenized corpus
    word_pairs = generate_word_pairs(tokenized_corpus, window_size)
    dataset = []
    logging.info(f"Length of word_pairs = {len(word_pairs)}")
    # Iterate over word pairs
    for target_word, context_word in word_pairs:
        logging.info("entered for loop")
        features = [0] * feature_count  # Initialize feature vector

        # Get index of target and context words from feature2id dictionary
        target_idx = feature2id.get(target_word, -1)
        context_idx = feature2id.get(context_word, -1)

        # If both target and context words are in the feature2id dictionary
        if target_idx != -1 and context_idx != -1:
            features[target_idx] = 1  # Set feature value to 1
            # Convert features to tensor
            tensor_features = torch.tensor(features)

            # Create tensor for label (context index)
            tensor_label = torch.tensor([context_idx], dtype=torch.long)

            # Append feature-label tuple to dataset
            dataset.append((tensor_features, tensor_label))

            # Log feature-label pair
            logging.info(
                f"Added feature-label pair to dataset: {tensor_features}, {tensor_label}")
        else:
            logging.warning(
                f"Target or context word not found in feature2id dictionary: Target={target_word}, Context={context_word}")

    logging.info("Dataset creation completed.")
    return dataset

#####################################################################################
# DEFINE NEURAL NETWORK ARCHITECTURES
#####################################################################################


class LogisticClassifier(nn.Module):

    def __init__(self, n_input, n_output):
        super(LogisticClassifier, self).__init__()

        self.layer1 = nn.Linear(n_input, n_output)

    def forward(self, inp):

        # Convert input tensor to the same data type as the weight tensor
        inp = inp.float()

        # Apply the linear layer
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
        # Convert input tensor to the same data type as the weight tensor
        inp = inp.float()

        # Convert the weight tensor of the first linear layer to torch.long
        self.layer1.weight = torch.nn.Parameter(self.layer1.weight.long())

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
        inp = inp.float()

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


def train(model, training_set, validation_set, test_set, eval_on_test=False, lr=0.001, n_epochs=1, batch_size=1, eval_every=1000, model_name="mlp"):
    logging.info("Training started...")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    training_examples_seen = 0
    best_loss = 1000000

    for epoch in range(n_epochs):

        for example_input, example_output in training_set:
            logging.info(
                f"Epoch {epoch}, Training examples seen: {training_examples_seen}")

            prediction = model(example_input)


            label = example_output

            logging.info(
                f"Prediction shape: {prediction.unsqueeze(0).shape}, Label shape: {label.shape}")
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
                logging.info(
                    f"Epoch {epoch}, Training examples seen: {training_examples_seen}, Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}")

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


    logging.info("Training completed.")
    logging.info("Evaluating on validation set...")
    logging.info(
        f"Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}")

    if eval_on_test:
        test_loss, test_accuracy = compute_evaluation_loss(
            model, test_set, loss_function)
        print("\nTEST METRICS:")
        print("Test loss:", test_loss, "Test accuracy:", test_accuracy)

    if eval_on_test:
        logging.info("Evaluating on test set...")
        test_loss, test_accuracy = compute_evaluation_loss(
            model, test_set, loss_function)
        logging.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", help="training file",
                        type=str, default="out/ppa.basic.train")
    parser.add_argument("--validation", "-v", help="validation file",
                        type=str, default="out/ppa.basic.dev")
    parser.add_argument("--test", "-e", help="test file",
                        type=str, default="out/ppa.basic.test")
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
    parser.add_argument("--batch_size", help="batch size", type=int, default=1)
    parser.add_argument(
        "--eval_every", help="this is the number n where we evaluate on the validation set after every n examples", type=int, default=5000)
    parser.add_argument(
        "--glove", help="Use concatenated GloVe embeddings as the input features", action='store_true')
    parser.add_argument(
        "--model", help="type of model (mlp2, mlp3, or logistic)", type=str, default="logistic")
    parser.add_argument("--random_seed", help="random seed",
                        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.glove:
        training_set = create_dataset_glove(args.train, label_count, label2id)
        validation_set = create_dataset_glove(
            args.validation, label_count, label2id)
        test_set = create_dataset_glove(args.test, label_count, label2id)

    else:

        feature2id, feature_count, label2id, label_count = feature_counter(
            args.train, args.validation, args.test) #TODO this is not working

        # Print out the values
        print("Feature to ID mapping:", feature2id)
        print("Total number of features:", feature_count)
        print("Label to ID mapping:", label2id)
        print("Total number of labels:", label_count)

        # tokenized_corpus = preprocess_and_generate_word_pairs(
        #     args.train, window_size)


        # 1. Preprocessing and Tokenization
        stripped_text = preprocess_text(posts)
        #print("Stripped Text:", stripped_text)
        corpus = twokenize.tokenize(stripped_text)
        #print("Corpus:", corpus)

        # 2. Word Pair Generation
        word_pairs = generate_word_pairs(corpus, window_size)
        #print("Word Pairs:", word_pairs)

        # 3. Indexing Words
        #print("Word to Index Dictionary:", word2idx)

        # 4. Creating Dataset
        dataset = create_dataset_from_tokens(
            corpus, window_size, feature_count, feature2id, label_count, label2id)
        print("Dataset:", dataset)
        
 

        training_set = create_dataset_from_tokens(
            corpus, window_size, feature_count, feature2id, label_count, label2id)
        validation_set = create_dataset_from_tokens(
            corpus, window_size, feature_count, feature2id, label_count, label2id)
        test_set = create_dataset_from_tokens(
            corpus, window_size, feature_count, feature2id, label_count, label2id)

    if args.model == "logistic":
        logging.info("using logistic model")
        model = LogisticClassifier(feature_count, label_count)
    elif args.model == "mlp2":
        logging.info("using mlp2 model")

        model = MLP2(feature_count, label_count,
                     hidden_size=args.hidden_size, dropout_p=args.dropout)
    elif args.model == "mlp3":
        logging.info("using mlp3 model")
        model = MLP3(feature_count, label_count,
                     hidden_size=args.hidden_size, dropout_p=args.dropout)
    train(model, training_set, validation_set, test_set, eval_on_test=args.eval_on_test, lr=args.lr,
          n_epochs=args.n_epochs, batch_size=args.batch_size, eval_every=args.eval_every, model_name="mlp")
