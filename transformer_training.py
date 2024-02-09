import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def get_next_word_probability(sentence, next_word):

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    #encoding words
    inputs = tokenizer.encode(sentence, return_tensors='pt')


    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()  

    # Predict all tokens
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    logits = outputs.logits


    next_word_logits = logits[0, -1, :]
    probabilities = torch.softmax(next_word_logits, dim=0)


    next_word_token_id = tokenizer.encode(next_word, add_prefix_space=True)[0]


    next_word_probability = probabilities[next_word_token_id].item()

    return next_word_probability


sentence = " Supreme Court has said that a defendant 's lawyer may take the position that the law requires."

liste = []
iterated = sentence.split()
for i, word in enumerate(iterated):
    if i == 0:
        continue
    liste.append(get_next_word_probability(" ".join(iterated[:i]),word))



print(liste)

print(sum(liste)/len(liste))
