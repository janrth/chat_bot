import random
import json
import pickle
import torch

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer

# import custome module:
from linear_nn import NeuralNet
#############################

# activate stemmer and device for pytroch:
stemmer = PorterStemmer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load json file with training data:
json_file_path = "intents.json"
with open(json_file_path, 'r') as j:
     contents = json.loads(j.read())

FILE = "data.pth"
data = torch.load(FILE)

# Find input size, size of hidden layers and output size + tags:
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
tags = data['tags']
model_state = data["model_state"]

# Instantiate pre-trained pytorch model:
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Load fitted vectorization for encoding:
vectorization_encoder = pickle.load(open('vectorization_encoder', 'rb'))

# Define bot name:
bot_name = "Yara"

# Start chat:
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    # stem + tokenize for input, before detokenization
    sentence = TreebankWordDetokenizer().detokenize([stemmer.stem(word) for word in 
                                                nltk.word_tokenize(sentence)])
    sentence = [sentence] # input sentence needs to be a list for vectorization
    X = vectorization_encoder.transform(sentence) # apply pre-fitted vectorization
    X = torch.from_numpy(X.toarray()).to(device) # from numpy to torch 

    output = model(X.float()) # make predict
    _, predicted = torch.max(output, dim=1) # return prediction with highest probability

    tag = tags[predicted.item()] # find correct tag

    probs = torch.softmax(output, dim=1) # find probability from softmax layer
    prob = probs[0][predicted.item()]  
    
    # If probability is big enough (so the bot is confident enough), then return an answer,
    # otherwise reply to repeat the question:
    if prob.item() > 0.6:
        for intent in contents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand, could you rephrase your answer, please. If we can not help you here, please visit our FAQ page at -insert_your_link_here-")