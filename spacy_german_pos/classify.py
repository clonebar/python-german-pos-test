import argparse

import spacy
import itertools


def iter_lines(file):
    return filter(
        lambda line: len(line) > 0,
        (line.strip('\n') for line in file)
    )


def iter_file_names(files):
    return (file.name for file in itertools.chain.from_iterable(files))


def print_result_short(doc):
    print([(token.text, token.tag_) for token in doc])


def print_result_full(doc):
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)


def print_result(doc, console_format):
    return {
        'short': print_result_short,
        'full': print_result_full
    }[console_format](doc)


def main(inputs, batch_size, console_format=None, quiet=False):
    nlp = spacy.load('de')
    for file in itertools.chain.from_iterable(inputs):
        if not quiet:
            print('---START: %s---' % file.name)
        docs = nlp.pipe(iter_lines(file), batch_size=batch_size)
        for doc in docs:
            if console_format:
                print_result(doc, console_format)
        if not quiet:
            print('---END---\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # main mandatory argument (-i)
    parser.add_argument('-i', '--inputs', type=argparse.FileType('r', encoding='utf-8'), nargs='+', action='append',
                        required=True, help='List of files to parse (multiple files allowed).')
    # additional optional arguments
    parser.add_argument('-b', '--batch', default=100, type=int,
                        help='Number of lines to buffer in an internal memory.', metavar='BATCH_SIZE')
    parser.add_argument('-p', '--print', default='short', type=str,
                        help='Print to console with given format argument (default is "short").', metavar='FORMAT')

    args = parser.parse_args()
    main(args.inputs, args.batch, args.print)
