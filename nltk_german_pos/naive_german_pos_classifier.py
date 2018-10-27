import re

from nltk import ClassifierBasedTagger


class GermanPosTagger(ClassifierBasedTagger):
    PATTERNS = {
        'number': re.compile('^\d+(,\d*)?|\d*,\d+$', re.UNICODE),
        'punct': re.compile('^\W+$', re.UNICODE),
        'upper-case': re.compile('^([a-zäöüß]+-)?[A-ZÄÖÜ]'),
        'lower-case': re.compile('^([a-zäöüß]+[/\'\-]?)+$'),
        'mixed-case': re.compile("\w+", re.UNICODE)
    }

    def __init__(self, train):
        self.counter = dict.fromkeys(GermanPosTagger.PATTERNS.keys(), 0)
        super().__init__(train=train)

    def feature_detector(self, tokens, index, history):
        word = tokens[index]
        word_lower = word.lower()
        features = {
            'word': word,
            'word-lower': word_lower,
            'prev-word': '<START>',
            'prev-prev-word': '<START>',
            'prev-tag': '<START>',
            'prev-prev-tag': '<START>',
            'shape': '<UNK>',
            'suffix(2)': word[-2:],
            'suffix(3)': word[-3:],
            'prefix(1)': word[1:]
        }

        if index > 0:
            features.update({
                'prev-word': tokens[index - 1],
                'prev-tag': history[index - 1],
            })
        if index > 1:
            features.update({
                'prev-prev-word': tokens[index - 2],
                'prev-prev-tag': history[index - 2]
            })

        for key, value in GermanPosTagger.PATTERNS.items():
            if value.match(word):
                features['shape'] = key
                self.counter[key] += 1
                break

        features.update({
            'prev-tag+word': '%s+%s' % (features['prev-tag'], word_lower),
            'prev-prev-tag+word': '%s+%s' % (features['prev-prev-tag'], word_lower),
            'prev-word+word': '%s+%s' % (features['prev-word'], word_lower),
        })

        return features
