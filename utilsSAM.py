from typing import List


def import_vocabulary(dataset) -> List[str]:
    default_voc = []
    for c in dataset:
        tmp = c["name"] + ", a"
        default_voc.append(c["name"].split(", ")[0])

    return default_voc


def take_vocabulary( dataset=None, add_words=None):

    if dataset is not None:
        vocabulary = import_vocabulary(dataset)

    if add_words is not None:
        add_words = list(set([v.lower().strip() for v in add_words]))
        # remove invalid vocabulary
        add_words = [v for v in add_words if v != ""]

        vocabulary = add_words + [c for c in vocabulary if c not in add_words]

    return vocabulary