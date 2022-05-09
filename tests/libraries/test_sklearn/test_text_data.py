"""Working with text data in sklearn.

* "Document": A single text object (tweet / article / comment / writing).
* "Corpus": A set of documents.

To download and unpack the training data for these examples (380MB total w/o
`unsup`):

wget -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -P data
tar xzf data/aclImdb_v1.tar.gz --skip-old-files -C data

# Remove unlabeled data (not used in these examples) rm -r
data/aclImdb/train/unsup

sklearn's `load_files` loads folders with the same folder structre as the test data

/
    train/
        neg/
        pos/
    test/
        neg/
        pos/

"""
