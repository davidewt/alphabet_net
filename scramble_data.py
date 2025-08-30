import json
import random

def load_alphabet_mapping():
    with open("data/alphabet_vectors.json", "r") as f:
        mapping = json.load(f)
    return mapping

def create_scrambled_test_data():
    mapping = load_alphabet_mapping()
    
    vectors = list(mapping.values())
    
    random.shuffle(vectors)
    
    return vectors

def save_test_data(vectors, filename="scrambled_test_vectors.json"):
    with open(f"data/{filename}", "w") as f:
        json.dump(vectors, f, indent=2)
        
if __name__ == "__main__":
    vectors = create_scrambled_test_data()
    save_test_data(vectors)
    print(f"Created scrambled test data with {len(vectors)} vectors")
    print("Saved to data/scrambled_test_vectors.json")