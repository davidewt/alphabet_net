# alphabet_net
A neural network that can recognize and learn the alphabet where each letter has been mapped to a randomized 4 dimensional vector

### Some things to know

- trained_alphabet_model.pth is trained on 1000 epochs
- trained_alphabet_model_1.pth is trained on 10,000 epochs

Since this is not real life data, the only data that exists is the one that has been generated.

generate_data.py generated a completely new alphabet-vector map

generate_graph.py visualises the alphabet vectors in a 3D-Color space. This is to see the similarities and vicinity of points. In the case that two point are very close to each other, there is a higher change to mix up the two.

- if you want to regenerate the corrupted data then just run corrupt_data.py again
- make sure to set the test_all_corrupted_levels() in the main.py and then run it
