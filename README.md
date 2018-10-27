Python tests for German Classifiers
===================================

This is a simple test repository for some German Part Of Speech (POS)
classifier. Be cautious. The project is only tested locally on Linux and
may contain several bugs. There is no working guarantee.

## Requirements

* Python 3
* Bash or some other shell

## Quick start

This quick start uses the python virtual environment feature to keep your
global installation clean. You may install the required packages globally
if you like. Though, a virtual environment is highly recommended.

### Commandline

Before you start, you have to launch a bash and
change to this repository's directory.

The following conventions apply for this quick start:

* Comments start with `#` and can be omitted
* The virtual environment is located under `./venv/`.

To initialize the project run the following commands in order:

1. Create a virtual environment:
    ```bash
    python -m venv ./venv/
    ```

2. Activate the virtual environment:
    ```bash
    source ./venv/bin/activate
    ```

3. Install required packages/data:

    ```bash
    # in active virtual environment (see step 2)

    # for nltk_german_pos run:
    pip install nltk
    python -m nltk.downloader punkt -d ./venv/nltk_data
    # end

    # for spacy_german_test run:
    pip install spacy
    python -m spacy download de
    # end
    ```

4. Download tiger corpus (only `nltk_german_pos`):

    * Go to their official download page (
    [Download Page](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/start.html)
    , [License Page](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/license/index.html)
    )
    * Download the `tigercorpus-2.2.conll09.tar.gz`
    * Unpack the content of `tigercorpus-2.2.conll09.tar.gz` into
    `./nltk_german_pos/data/`. There should be only one file named
    `tiger_release_aug07.corrected.16012013.conll09`.

Afterwards you can execute the following commands:

> **Hint:** Swap the placeholder `$INPUT_FILE` with your own text corpus in the commands.

* `nltk_german_pos`:

    > **Warning:**
    >
    > The first time you run the classifier, it may seem very slow.
    > Don't worry. It's normal. This test does not use any pre-trained model.
    > So it needs to build its model from scratch. This is done by using the
    > features in `./nltk_german_pos/naive_german_pos_classifer.py`.

    * To list all options:

        ```bash
        python ./nltk_german_pos/classify.py --help
        ```

    * To run the classifier:

        ```bash
        python ./nltk_german_pos/classify.py -i $INPUT_FILE
        ```

    * To (re-)train the classifier:

        ```bash
        python ./nltk_german_pos/classify.py -r
        ```

    * To evaluate the classifier on random test data from the tiger corpus:

        ```bash
        python ./nltk_german_pos/classify.py -e
        ```

        > **Hint:** optional use `-s $SEED` to set the random generator seed

* `spacy_german_pos`:

    * To list all options:

        ```bash
        python ./spacy_german_pos/classify.py --help
        ```

    * To run the classifier:

        ```bash
        python ./spacy_german_pos/classify.py -i $INPUT_FILE
        ```

    * To run with detailed output:

        ```bash
        python ./spacy_german_pos/classify.py -i $INPUT_FILE -p full
        ```
