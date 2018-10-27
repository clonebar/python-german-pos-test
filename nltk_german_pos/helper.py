import os
import pickle
import random
import sys
from pathlib import Path

import nltk

SCRIPT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = SCRIPT_PATH.joinpath('data')


def shuffle(tagged, rng):
    indices = rng.sample(range(len(tagged)), len(tagged))
    return [tagged[i] for i in indices]


def generate_seed(seed=None):
    if seed is None:
        seed = random.randrange(sys.maxsize)
    return seed


def read_tiger(path, root=DATA_PATH):
    return nltk.corpus.ConllCorpusReader(str(root), path,
                                         ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                         encoding='utf-8')


def load_tagger(path, root=DATA_PATH):
    file = root.joinpath(path)
    if not file.is_file():
        return None
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_tagger(path, tagger, root=DATA_PATH):
    file = root.joinpath(path)
    with open(file, 'wb') as f:
        pickle.dump(tagger, f, protocol=2)


def iter_lines(file):
    return filter(
        lambda line: len(line) > 0,
        (line.strip('\n') for line in file)
    )
