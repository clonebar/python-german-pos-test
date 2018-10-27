import argparse
import itertools
import random

import helper
import nltk
from naive_german_pos_classifier import GermanPosTagger

DATA_FILE = 'nltk_german_pos_data.pickle'
TIGER_FILE = 'tiger_release_aug07.corrected.16012013.conll09'


def init_tagged(percentage, rng):
    corp = helper.read_tiger(TIGER_FILE)
    tagged_sents = corp.tagged_sents()
    return helper.shuffle(tagged_sents, rng), int(len(tagged_sents) * percentage)


def run_evaluation(tagger, test):
    accuracy = tagger.evaluate(test)
    print('accuracy:', accuracy)


def evaluate_seed(seed):
    if seed is None or seed < 0:
        seed = helper.generate_seed()
    rng = random.Random(seed)
    print('seed: ', seed)
    return seed, rng


def main(inputs=(), evaluate=False, retrain=False, seed=None):
    contents = []
    for file in itertools.chain.from_iterable(inputs):
        contents += helper.iter_lines(file)
    print('input:', contents)

    seed, rng = evaluate_seed(seed)

    percentage = 0.1
    tagged = None
    split = None
    tagger = None

    if not retrain:
        tagger = helper.load_tagger(DATA_FILE)
    if tagger is None:
        print('preparing data for training...')
        if tagged is None:
            tagged, split = init_tagged(percentage, rng)
        print('train size:', len(tagged) - split, '/', len(tagged))

        print('training...')
        tagger = GermanPosTagger(tagged[split:])
        print(tagger.counter)
        helper.save_tagger(DATA_FILE, tagger)

    if evaluate:
        print('preparing data for evaluation...')
        if tagged is None:
            tagged, split = init_tagged(percentage, rng)
        print('test size:', split, '/', len(tagged))
        print('evaluating...')
        run_evaluation(tagger, tagged[:split])

    contents = [nltk.word_tokenize(content, 'german') for content in contents]
    for i, content in enumerate(contents):
        tags = tagger.tag(content)
        print('tagged:', tags)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', type=argparse.FileType('r', encoding='utf-8'), nargs='+', action='append',
                        help='List of files to parse (multiple files allowed).')
    parser.add_argument('-e', '--evaluate', action='store_true',
                        help='Calculates the accuracy on a random test set.')
    parser.add_argument('-r', '--retrain', action='store_true',
                        help='Trains the model again regardless of any previous run.')
    parser.add_argument('-s', '--seed', type=int,
                        help='Sets a fixed seed value for the random generator.')
    args = parser.parse_args()
    main(args.inputs, args.evaluate, args.retrain, args.seed)
