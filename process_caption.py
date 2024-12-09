import nltk

from itertools import chain
from utils.utilsSAM import read_pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Download stopwords and tokenizer models if not already done
# nltk.download('stopwords')
# nltk.download('punkt_tab')

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

    print(processed_captions)
    return processed_captions

if __name__ == '__main__':
    
    captions = read_pickle(PATH_CAPTION)

    captions_filtered = filter_caption(captions)
    captions_filtered = set(chain(*captions_filtered))

    print("-"*90)
    print(captions_filtered)


    


    



