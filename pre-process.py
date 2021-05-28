# preprocess.py should take the training (or test) directory containing movie reviews,
# should perform pre-processing (noise removal, tokenization, and word count) on each file 
# and output the files in the vector format to be used by NB.py.

import os
import sys
import json

def remove_noise(text):
    # Remove punctuation/special characters and lowercase text
    new_text = ""
    special_characters = {'"', '*', '+', '.', '_', '`', ',', '<', '>', '@', '^', '{', '}', '|', '~'}
    for char in text:
        if char not in special_characters:
            new_text += char.lower()
    return new_text

def tokenize(text):
    return text.split()

def ignoreunseen(words, vocab):
    # Disregard any words not specified in .vocab file
    return [word for word in words if word in vocab]

def word_count(text): 
    # Stores each word occurence in a dict
    frequencies = {}
    for word in text:
        if word in frequencies:
            frequencies[word] += 1 # If that word exists in there, add one to frequency
        else:
            frequencies[word] = 1 # First time discovering word
    return frequencies

def preprocess():
    feature_vectors = []
    for label in os.listdir(directory): 
        folder = os.path.join(directory, label)
        if os.path.isdir(folder):
            for filename in os.listdir(folder):
                if filename.endswith(".txt"):
                    f = open(os.path.join(folder, filename), "r")
                    words = remove_noise(f.read())
                    words = tokenize(words)
                    words = ignoreunseen(words, vocab)
                    feature_vectors.append({label: word_count(words)})
                    f.close()

    output_name = directory.replace("/", "").replace("reviews", "reviews-BOW-") + ".NB"
    output = open(output_name, "w")
    for line in feature_vectors:
        output.write(json.dumps(line) + '\n')
    output.close()


directory = sys.argv[1]
vocab = set([line.rstrip() for line in open('reviews/imdb.vocab')])
preprocess()

