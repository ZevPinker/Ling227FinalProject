#processing raw training text into a word2vec model

import re
import pandas as pd
from string import ascii_lowercase, ascii_uppercase, digits
import twokenize


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from itertools import combinations
import random

# Function to generate word pairs from tokenized corpus


def generate_word_pairs(corpus, window_size):
    word_pairs = []
    for i, target_word in enumerate(corpus):
        context = corpus[max(0, i - window_size):i] + \
            corpus[i + 1:i + window_size + 1]
        for context_word in context:
            word_pairs.append((target_word, context_word))
    return word_pairs

# Function to train Skip-gram Word2Vec model


# Function to train Skip-gram Word2Vec model
def train_word2vec(corpus, vocabulary, vector_size=100, min_count=1, window=5, epochs=10):
    # Initialize Word2Vec model
    model = Word2Vec(vector_size=vector_size, min_count=min_count)
    model.build_vocab_from_freq({word: 0 for word in vocabulary})

    # Build vocabulary from the corpus
    model.build_vocab(corpus_iterable = corpus)


    # Print the vocabulary
    # 'index_to_key' contains the list of all unique tokens in the model's vocabulary
    vocab = list(model.wv.index_to_key)
    print("Vocabulary size:", len(vocab))
    # Print first 50 words in the vocabulary
    print("Sample vocabulary:", vocab[:50])




    # Train Word2Vec model
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs)

    return model




def remove_duplicates(lst):
    return list(dict.fromkeys(lst))


df = pd.read_csv('data/data.csv')

posts = df['text'][0]

stripped = ""

#loops through text and removes punctuation and lowercases all letters. 
for char in posts:
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

corpus = twokenize.tokenize(stripped)
print(corpus[0:100])
vocabulary = set(corpus)
print(vocabulary)


# Hyperparameters
window_size = 2
vector_size = 100
min_count = 1
epochs = 10

# Train Word2Vec model
word2vec_model = train_word2vec(
    corpus, vocabulary, window_size, vector_size, min_count, epochs)

# Print most similar words to 'fox' as an example
print("Most similar words to 'me':", word2vec_model.wv.most_similar('me'))

#print(tokens[0:50])
#print(string_with_no_punctuation[0:100])



