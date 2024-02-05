import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def get_next_word_probability(sentence, next_word):

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

 
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


sentence = "The quick brown fox jumps"
next_word = "over"
probability = get_next_word_probability(sentence, next_word)
print(f"Probability of '{next_word}' being the next word: {probability}")
