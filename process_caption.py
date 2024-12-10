import nltk
import string

from itertools import chain
from utils.utilsSAM import read_pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag


# Download stopwords and tokenizer models if not already done
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger_eng')

PATH_CAPTION = "datasets/captions_val/captions.pkl"

def filter_caption(captions: str) -> str:
    
    # Load English stopwords
    stop_words = set(stopwords.words('english'))

    # Process captions
    processed_captions = []
    for caption in captions:
        # Tokenize the caption
        words = word_tokenize(caption.lower())
        # Remove stopwords and punctuation
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        processed_captions.append(filtered_words)
    return processed_captions

def extract_noun_phrases(text):
    
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]
    tagged = pos_tag(tokens)
    # print(tagged)
    grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}'
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged)
    
    nouns = []
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            nouns.append(' '.join(t[0] for t in subtree.leaves() if t[1] == 'NN' and t[0] != None))   #extract only the noun part 
    
    return set(nouns)


if __name__ == '__main__':
    
    captions = read_pickle(PATH_CAPTION)

    text = " ".join(captions)
    nouns = extract_noun_phrases(text)
    print(nouns, len(nouns))

    captions_filtered = filter_caption(captions)
    captions_filtered = set(chain(*captions_filtered))

    print("-"*90)
    print(len(captions_filtered))



