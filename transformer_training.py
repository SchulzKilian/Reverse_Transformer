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


sentence = " In my position I have now, about half of my time is devoted to counseling and registration and other issues like that. About thirty to forty percent of my time is involved with teaching, doing preparation, helping out in the labs, and helping students. About five to ten percent of my time is spent being involved in academic committees and working with administrative items."

liste = []
iterated = sentence.split()
for i, word in enumerate(iterated):
    if i == 0:
        continue
    liste.append(get_next_word_probability(" ".join(iterated[:i]),word))



print(liste)

print(sum(liste)/len(liste))
