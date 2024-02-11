import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def get_next_word_probability(sentence, next_word):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # words are encoded here
    inputs = tokenizer.encode(sentence, return_tensors='pt')

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()  

    # tokens are predicted here
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    logits = outputs.logits

    next_word_logits = logits[0, -1, :]
    probabilities = torch.softmax(next_word_logits, dim=0)

    next_word_token_id = tokenizer.encode(next_word, add_prefix_space=True)[0]
    next_word_probability = probabilities[next_word_token_id].item()

    return next_word_probability

# sentences from the file are processed here
def process_sentences(input_file, output_file):
    with open(input_file, 'r') as f:
        sentences = f.readlines()

    with open(output_file, 'w') as f_out:
        for sentence in sentences:
            sentence = sentence.strip()
            liste = []
            iterated = sentence.split()
            for i, word in enumerate(iterated):
                if i == 0:
                    continue
                liste.append(get_next_word_probability(" ".join(iterated[:i]), word))
            avg_probability = sum(liste) / len(liste)
            f_out.write(f'"{sentence}","{liste}","{avg_probability}"\n')

# Define input and output file paths
input_file = '/Users/zoe/Documents/GitHub/zoematr/Reverse_Transformer/data/HU_sentences.txt'
output_file = '/Users/zoe/Documents/GitHub/zoematr/Reverse_Transformer/data/HU_sentences_results.txt'

# Process sentences and write results to the output file
process_sentences(input_file, output_file)
