import json
import random
import copy

def load_alphabet_mapping():
    with open("data/alphabet_vectors.json", "r") as f:
        mapping = json.load(f)
    return mapping

def corrupt_vector(vector, corruption_level=1):
    corrupted = vector.copy()
    
    num_elements_to_corrupt = random.randint(1, corruption_level)
    indices_to_corrupt = random.sample(range(len(vector)), num_elements_to_corrupt)
    
    for i in indices_to_corrupt:
        change = random.randint(-corruption_level, corruption_level)
        if change == 0:
            change = random.choice([-1,1])
        
        new_value = max(0, min(9, corrupted[i] + change))
        corrupted[i] = new_value
    
    return corrupted

def create_corrupted_dataset(corruption_level=1):
    mapping = load_alphabet_mapping()
    corrupted_data = {}
    
    for letter, vector in mapping.items():
        corrupted_vector = corrupt_vector(vector, corruption_level)
        corrupted_data[letter] = corrupted_vector
        
    return corrupted_data

def save_corrupted_data(corrupted_data, level):
    filename = f"data/corrupted_data_level_{level}.json"
    with open(filename, "w") as f:
        json.dump(corrupted_data, f, indent=2)
    
    return filename

if __name__ == "__main__":
    for level in [1,2,3,4]:
        corrupted_data = create_corrupted_dataset(level)
        filename = save_corrupted_data(corrupted_data, level)
        print(f"created corruption level {level}: {filename}")
    