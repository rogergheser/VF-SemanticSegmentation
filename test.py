import pickle

with open('datasets/ADE20K_2021_17_01/captions_val/vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

print(vocabulary)
print(len(vocabulary))