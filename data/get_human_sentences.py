import nltk
import random

# Download the Gutenberg corpus if not already downloaded
nltk.download('gutenberg')
nltk.download('punkt')

# Load Gutenberg corpus
from nltk.corpus import gutenberg


# Get a list of sentences from Gutenberg corpus
sentences = gutenberg.sents()

# Filter sentences based on length
filtered_sentences = [sent for sent in sentences if 20 <= len(sent) <= 50]

# Shuffle the sentences
random.shuffle(filtered_sentences)

# Select 400 sentences (or less if there aren't enough available)
selected_sentences = filtered_sentences[:400]

# Write selected sentences to a text file
with open("/Users/zoe/Documents/GitHub/zoematr/Reverse_Transformer/data/HU_sentences.txt", "w") as file:
    for sent in selected_sentences:
        # Join words in the sentence and write to the file
        file.write(" ".join(sent) + "\n")
