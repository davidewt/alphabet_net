import json
import random
import os
import string

alphabet_list = list(string.ascii_lowercase)
alphabet_mapping = {}

test_vector = [0,1,2,3]
for letter in alphabet_list:
    while True:
        vector = [random.randint(0,9),random.randint(0,9),random.randint(0,9),random.randint(0,9)]
        if vector not in alphabet_mapping.values():
            alphabet_mapping[letter] = vector
            break

print(alphabet_mapping)

if not os.path.exists("data"):
    os.makedirs("data")
    
with open("data/alphabet_vectors.json", "w") as f:
    json.dump(alphabet_mapping, f, indent=2)


