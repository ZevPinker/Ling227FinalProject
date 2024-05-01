import re
import pandas as pd
from string import ascii_lowercase, ascii_uppercase, digits
import twokenize

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

tokens = twokenize.tokenize(stripped)
print(tokens[0:50])
#print(string_with_no_punctuation[0:100])
