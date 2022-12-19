import numpy as np
import nltk 
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch.utils.data import Dataset, DataLoader
import json
import pickle

import data_set
import linear_nn

#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
##########################

def detokenize_fct(x):
    '''
    Applies detokanization to obtain full sentences.
    '''
    return TreebankWordDetokenizer().detokenize(x)

# Instantiate stemmer and count-vectorizer:
stemmer = PorterStemmer()
vectorizer = CountVectorizer()

# load json file with training data:
json_file_path = "intents.json"
with open(json_file_path, 'r') as j:
     contents = json.loads(j.read())


# Training begins here
# Create empty lists to save tags and full sentences 
# for stemming and tokenization:
tags = []
sentences = []

# loop through each sentence in training data:
for intent in contents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        # tokenization and stemming applied to each word, before detokenized sentence is saved
        sentences.append(detokenize_fct([stemmer.stem(word) for word in nltk.word_tokenize(pattern)]))
        tags.append(tag)

# The detokanization is needed as the vectorizer is applied on one list of many sentences:
X_train = vectorizer.fit_transform(sentences)

# Create unique tags and numeric class labels: 
tags_unique = sorted(set(tags)) 
y_train = [tags_unique.index(i) for i in tags] 

# save vectorizer:
pickle.dump(vectorizer, open('vectorization_encoder', 'wb'))

# Train model:
X_train = X_train.toarray()
y_train = np.array(y_train)

# Hyper-parameters for pytroch NeuralNet: 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 500
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

# Create dataset to be consumed by the model + instantiate the Neural Network:
dataset = data_set.ChatDataset(X_train, y_train)
model = linear_nn.NeuralNet(input_size, hidden_size, output_size).to(device)

train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(words.float())
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

# Save the model:
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"tags": tags_unique
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')