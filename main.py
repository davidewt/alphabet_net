import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def load_alphabet_mapping():
    """Load the alphabet-to-vector mapping from JSON file"""
    with open("data/alphabet_vectors.json", "r") as f:
        mapping = json.load(f)
    return mapping

def prepare_training_data():
    mapping = load_alphabet_mapping()
    vectors = []
    letter_numbers = []
    alphabet_list = list(string.ascii_lowercase)
    
    # Fill the lists from the mapping
    for letter, vector in mapping.items():
        vectors.append(vector)
        letter_number = alphabet_list.index(letter)
        letter_numbers.append(letter_number)
        
    X = torch.tensor(vectors, dtype=torch.float32)
    y = torch.tensor(letter_numbers, dtype=torch.long)
    
    return X, y

def setup_training():
        model = AlphabetNet()
        
        criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        return model, criterion, optimizer
    
def train_model():
    X, y = prepare_training_data()
    model, criterion, optimizer = setup_training()
    
    num_epochs = 10000
    for epoch in range(num_epochs):
        
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.7f}')
    
    torch.save(model.state_dict(), 'trained_alphabet_model_1.pth')
    print("Model saved!")
    
    return model

def load_trained_model():
    model = AlphabetNet()
    model.load_state_dict(torch.load('trained_alphabet_model_1.pth'))
    model.eval()  # Set to evaluation mode
    return model

def test_model():
    model = load_trained_model()
    mapping = load_alphabet_mapping()
    
    for letter, vector in list(mapping.items()):
        input_tensor = torch.tensor([vector], dtype=torch.float32)
        output = model(input_tensor)
        predicted_letter_num = torch.argmax(output).item()
        predicted_letter = chr(predicted_letter_num + ord('a'))
        
        # Getting probability
        probabilities = F.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()
        
        print(f"Vector {vector} --> Predicted: {predicted_letter} | {confidence} |, Actual: {letter}")
        
def load_scrambled_test_data():
    with open("data/scrambled_test_vectors.json", "r") as f:
        vectors = json.load(f)
    return vectors

def test_model_on_unkown_data():
    model = load_trained_model()
    test_vectors = load_scrambled_test_data()
    
    print("-" * 30)
    
    for i, vector in enumerate(test_vectors):
        input_tensor = torch.tensor([vector], dtype=torch.float32)
        output = model(input_tensor)
        
        probabilities = F.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()
        predicted_letter_num = torch.argmax(output).item()
        predicted_letter = chr(predicted_letter_num + ord('a'))
        
        print(f"Test {i+1}: Vector {vector} --> Pred: {predicted_letter} | Conf: {confidence:.3f}")

def load_corrupted_data(level):
    """Load corrupted data for a specific corruption level"""
    filename = f"data/corrupted_data_level_{level}.json"
    with open(filename, "r") as f:
        corrupted_mapping = json.load(f)
    return corrupted_mapping

def test_model_on_currupted_data(level):
    model = load_trained_model()
    corrupted_mapping = load_corrupted_data(level)
    
    print(f"\nTesting on Corruption Level {level}")
    print("-" * 50)
    
    correct_predictions = 0
    total_predictions = 0
    
    for actual_letter, corrupted_vector in corrupted_mapping.items():
        input_tensor = torch.tensor([corrupted_vector], dtype=torch.float32)
        output = model(input_tensor)
        
        probabilities = F.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()
        predicted_letter_num = torch.argmax(output).item()
        predicted_letter = chr(predicted_letter_num + ord('a'))
        
        is_correct = predicted_letter == actual_letter
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        status = "Y" if is_correct else "N"
        print(f"{status} {corrupted_vector} --> Pred: {predicted_letter} | Act: {actual_letter} | Conf: {confidence:.3f}")
        
    accuracy = correct_predictions / total_predictions * 100
    print(f"\n Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.1f}%")
    return accuracy

def test_all_corrupted_levels():
    print("Testing Model Robustness Against Corruption")
    print("=" * 60)
    
    accuracies = {}
    
    for level in [1,2,3,4]:
        accuracy = test_model_on_currupted_data(level)
        accuracies[level] = accuracy
        
    print("\n" + "=" * 60)
    print("SUMMARY:")
    for level, accuracy in accuracies.items():
        print(f"Corruption Level {level}: {accuracy:.1f}% accuracy")

class AlphabetNet(nn.Module):
    def __init__(self):
        super(AlphabetNet, self).__init__()
        
        self.fc1 = nn.Linear(4, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,32)
        self.fc4 = nn.Linear(32,26)
        
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
if __name__ == "__main__":
    # trained_model = train_model()
    # print("Training Complete")
    # test_model()
    # test_model_on_unkown_data()
    test_all_corrupted_levels()
