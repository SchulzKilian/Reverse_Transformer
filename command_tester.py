import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(50, 100)  # Input layer with 50 features, outputting to hidden layer with 100 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 50)  # Another hidden layer
        self.fc3 = nn.Linear(50, 1)  # Output layer, outputting a single value
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid to ensure output is between 0 and 1
        return x




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

# Instantiate the model
model_loaded = SimpleNN()

# Load the saved state dict into the model
model_loaded.load_state_dict(torch.load('simple_nn_model.pt'))

# Make sure to call eval() to set dropout and batch normalization layers to evaluation mode
model_loaded.eval()

while True:
    sentence = input("Input your sentence here\n")

    liste = []
    iterated = sentence.split()
    for i, word in enumerate(iterated):
        if i == 0:
            continue
        liste.append(get_next_word_probability(" ".join(iterated[:i]),word))
    for k in range(50-len(liste)):    #flattening the vector to 50
        liste.append(0.0)

    input_tensor = torch.tensor([liste], dtype=torch.float32) 



    
    with torch.no_grad():  # No need to compute gradients for inference
        output = model_loaded(input_tensor)
        print("\n")
        print("The probability that you are a human according to this quite limited sample size is: \n")
        print(str(output.item())+" percent.\n")
        print("Please remember this classifier has been trained on very simple GPT 2 data and can not be applied to state of the art AI models.\n")