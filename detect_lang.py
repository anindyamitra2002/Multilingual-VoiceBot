import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.autograd import Variable
import argparse
from glob import glob
import matplotlib.pyplot as plt
from LIDv2.ccc_wav2vec_extractor import HiddenFeatureExtractor

# Number of language classes 
Nc = 12
look_back1 = 20 
look_back2 = 50
IP_dim = 1024  # number of input dimension

# Function to return processed input data (feature/vector)
def lstm_data(npy_path):
    X = npy_path
    Xdata1, Xdata2 = [], []
    mu, std = X.mean(axis=0), X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std

    for i in range(0, len(X) - look_back1, 1):  # High resolution low context
        a = X[i:(i + look_back1), :]
        Xdata1.append(a)
    for i in range(0, len(X) - look_back2, 2):  # Low resolution long context
        b = X[i + 1:(i + look_back2):3, :]
        Xdata2.append(b)

    Xdata1 = torch.from_numpy(np.array(Xdata1)).float()
    Xdata2 = torch.from_numpy(np.array(Xdata2)).float()

    return Xdata1, Xdata2

# Define the LSTMNet model
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(1024, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * 256, 32, bidirectional=True)
        self.fc_ha = nn.Linear(2 * 32, 100)
        self.fc_1 = nn.Linear(100, 1)
        self.sftmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x)
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht = torch.unsqueeze(ht, 0)
        ha = torch.tanh(self.fc_ha(ht))
        alp = self.fc_1(ha)
        al = self.sftmax(alp)
        T = ht.size(1)
        batch_size = ht.size(0)
        D = ht.size(2)
        c = torch.bmm(al.view(batch_size, 1, T), ht.view(batch_size, T, D))
        c = torch.squeeze(c, 0)
        return c

# Define the MSA_DAT_Net model
class MSA_DAT_Net(nn.Module):
    def __init__(self, model1, model2):
        super(MSA_DAT_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.att1 = nn.Linear(2 * 32, 100)
        self.att2 = nn.Linear(100, 1)
        self.bsftmax = nn.Softmax(dim=1)
        self.lang_classifier = nn.Linear(2 * 32, Nc, bias=True)

    def forward(self, x1, x2):
        u1 = self.model1(x1)
        u2 = self.model2(x2)
        ht_u = torch.cat((u1, u2), dim=0)
        ht_u = torch.unsqueeze(ht_u, 0)
        ha_u = torch.tanh(self.att1(ht_u))
        alp = torch.tanh(self.att2(ha_u))
        al = self.bsftmax(alp)
        Tb = ht_u.size(1)
        batch_size = ht_u.size(0)
        D = ht_u.size(2)
        u_vec = torch.bmm(al.view(batch_size, 1, Tb), ht_u.view(batch_size, Tb, D))
        u_vec = torch.squeeze(u_vec, 0)
        lang_output = self.lang_classifier(u_vec)
        return lang_output, u1, u2, u_vec

# New function to initialize the model and evaluator
def initialize_model():
    model1 = LSTMNet()
    model2 = LSTMNet()
    model = MSA_DAT_Net(model1, model2)
    model_path = './LIDv2/model/ZWSSL_train_SpringData_13June2024_e3.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    
    # Initialize HiddenFeatureExtractor
    evaluator = HiddenFeatureExtractor()
    
    return model, evaluator

# New function to process a wav file and return the predicted language
def predict_language_wav(model, evaluator, wav_path):
    file_names, speech_list = evaluator.preprocess_audio([wav_path])

    if len(speech_list[0]) <= 16400:
        print("Error: Audio file too short for classification.")
        return None
    
    hidden_features = evaluator.hiddenFeatures([speech_list[0]])
    X1, X2 = lstm_data(hidden_features[0])
    X1 = np.swapaxes(X1, 0, 1)
    X2 = np.swapaxes(X2, 0, 1)
    
    x1 = Variable(X1, requires_grad=False)
    x2 = Variable(X2, requires_grad=False)
    
    lang_output, _, _, _ = model.forward(x1, x2)
    output = lang_output.detach().cpu().numpy()[0]
    pred_all = np.exp(output) / np.sum(np.exp(output))
    Pred = np.argmax(output)

    # Language mapping
    id2lang = {0: 'asm', 1: 'ben', 2: 'eng', 3: 'guj', 4: 'hin', 5: 'kan', 6: 'mal', 7: 'mar', 8: 'odi', 9: 'pun', 10: 'tam', 11: 'tel'}
    predicted_language = id2lang[Pred]

    return predicted_language, pred_all

def main():
    parser = argparse.ArgumentParser(description='Spoken language identification (WSSL uVector) script with command line options.')
    parser.add_argument('path', help='Path to the .wav file')
    args = parser.parse_args()
    path = args.path

    # Ensure the input path is a valid .wav file
    if not os.path.isfile(path) or not path.endswith(".wav"):
        print(f"Error: {path} is not a valid .wav file.")
        return

    # Initialize the model and evaluator
    model, evaluator = initialize_model()

    # Predict the language of the given .wav file
    predicted_language, probabilities = predict_language_wav(model, evaluator, path)

    if predicted_language:
        print(f"The predicted language for the audio file '{path}' is {predicted_language}")

# if __name__ == "__main__":
#     main()
