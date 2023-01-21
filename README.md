Basic chat bot for the purpose of holiday booking/information for hotels. 

A json file contains the intents, which are used during training. Stemming and bag-of-words is applied to decode 
the sentences. A simple feed forward Neural Network using pytorch is utilized for training. 
The chat bot is activated via a python file. If the soft-max probability is below a threshold the chat bot will 
not return the learned answer, but will ask to rephrase the question.

To train the model use the terminal and type:
- python train_chatbot.py

To start the trained bot for interaction use the terminal and type:
- python chat.py
