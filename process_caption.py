import nltk
import string
import pickle

from transformers import BertTokenizer, BertModel
import torch

from tqdm import tqdm
from scipy.spatial.distance import cosine
from datasets.dataset_vars import ADE20K_SEM_SEG_FULL_CATEGORIES as ADE20K_CATEGORIES
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
            nouns.append(' '.join(t[0] for t in subtree.leaves() if t[1] == 'NN' and t[0] != ""))   #extract only the noun part 
    
    return set(nouns)

def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

# Compute cosine similarity
def compute_similarity(embedding1, embedding2):
    similarity = 1 - cosine(embedding1.numpy(), embedding2.numpy())
    return similarity

def get_ade_dict(ade_gt, tokenizer, model):
    ade_encoding = {}
    ade_name_to_id = {}
    ade_id_to_name = {}
    for sample in ade_gt:
        name, id, trainId = sample['name'], sample['id'], sample['trainId']
        ade_name_to_id[name] = trainId
        ade_id_to_name[trainId] = name
        ade_encoding[name] = encode_text(name, tokenizer, model)

    # Sort the encoding by trainId
    sorted_ade_encoding = sorted(ade_encoding.items(), key=lambda item: ade_name_to_id[item[0]])

    # Convert sorted_ade_encoding to a tensor
    ade_encoding_tensor = torch.stack([item[1] for item in sorted_ade_encoding])

    return ade_encoding_tensor, ade_name_to_id, ade_id_to_name

def update_old_vocab(ade_gt, new_voc):
    # ade_gt list[dict] --> with keys (name, id, trainId)
    # Load the model
    tokenizer, model = load_bert_model()
    # Encode the old vocab and store it in a dictionary
    # Ordered list of the old vocab by trainId
    ade_vocab = sorted(ade_gt, key=lambda x: x['trainId'])
    # Encode the old vocab
    encoded_ade = encode_word_list([sample['name'] for sample in ade_vocab], tokenizer, model)

    # Encode the new vocab in batches of BATCH_SIZE
    BATCH_SIZE = 64
    new_voc = list(new_voc)
    # Init with -2 to indicate its the original word
    ret_voc = {int(sample['trainId']) : (sample['name'], -2) for sample in ade_vocab}

    new_voc_batches = [new_voc[i:i + BATCH_SIZE] for i in range(0, len(new_voc), BATCH_SIZE)]

    for batch in tqdm(new_voc_batches, desc="Processing new vocab", total=len(new_voc_batches)):
        new_voc_embeddings = (encode_word_list(batch, tokenizer, model))
        # Compute the similarity between the new vocab and the old vocab
        similarity = (new_voc_embeddings @ encoded_ade.T)
        # Find the most similar old vocab for each new vocab
        max_similarity, max_indices = similarity.topk(1, dim=1)
        for i, (sim, idx) in enumerate(zip(max_similarity, max_indices)):
            idx = idx.item()
            # if sim > 0.5:
            if ret_voc[idx][1] < sim.item():
                ret_voc[int(idx)] = (batch[i], sim.item())

    # Substitute all sim -2 with 1
    unchanged_words = 0
    for key, value in ret_voc.items():
        if value[1] == -2:
            ret_voc[key] = (value[0], (value[0], 1))
            unchanged_words += 1
        if ret_voc[key][0] == ade_gt[key]['name']:
            ret_voc[key] = (ret_voc[key][0], 1)
            unchanged_words += 1

    print(f"Unchanged words: {unchanged_words}")

    return ret_voc

def encode_word_list(word_list, tokenizer, model):
    inputs = tokenizer(word_list, return_tensors="pt", padding=True, truncation=True, max_length=128, add_special_tokens=True)

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token embedding for each word
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
    return cls_embeddings / torch.norm(cls_embeddings, dim=1, keepdim=True)

def strip_noun(noun):
    if 'a ' in noun:
        noun = noun.replace('a ', '')
    elif 'an ' in noun:
        noun = noun.replace('an ', '')
    elif 'the ' in noun:
        noun = noun.replace('the ', '')
    return noun

if __name__ == '__main__':
    
    captions = read_pickle(PATH_CAPTION)

    text = " ".join(captions)
    nouns = extract_noun_phrases(text)
    print(list(nouns), len(nouns))
    # save captions in a pickle file 
    with open("datasets/captions_val/nouns_ade20k.pkl", "wb") as f:
        pickle.dump(list(nouns), f)


    nouns = [strip_noun(noun) for noun in nouns]
    print(sorted(list(set(nouns))))

    new_vocab = update_old_vocab(ADE20K_CATEGORIES, nouns)
    print(new_vocab)

    captions_filtered = filter_caption(captions)
    captions_filtered = set(chain(*captions_filtered))

    print("-"*90)
    print(len(captions_filtered))



