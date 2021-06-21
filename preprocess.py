import argparse
import os
import csv
import numpy as np
import tensorflow as tf
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from nltk.stem.wordnet import WordNetLemmatizer
import model.data

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

csv.field_size_limit(2**28)

def clean_text(sentence):
    # remove non alphabetic sequences
    pattern = re.compile(r'[^a-z]+')
    sentence = sentence.lower()
    sentence = pattern.sub(' ', sentence).strip()
    
    # Tokenize
    word_list = word_tokenize(sentence)
    
    # stop words
    stopwords_list = set(stopwords.words('english'))
    # puctuation
    punct = set(string.punctuation)
    
    # remove stop words
    word_list = [word for word in word_list if word not in stopwords_list]
    # remove very small words, length < 3
    word_list = [word for word in word_list if len(word) > 2]
    # remove punctuation
    word_list = [word for word in word_list if word not in punct]
    
    # stemming
    # ps  = PorterStemmer()
    # word_list = [ps.stem(word) for word in word_list]
    
    # lemmatize
    lemma = WordNetLemmatizer()
    word_list = [lemma.lemmatize(word) for word in word_list]

    # list to sentence
    sentence = ' '.join(word_list)
    
    return word_list #, sentence 

def tokens(text):
    return [w.lower() for w in clean_text(text)]

def preprocess_tf(text, vocab_to_id):
    ids = [vocab_to_id.get(x) for x in tokens(text) if vocab_to_id.get(x)]
    if ids:
        vector = np.bincount(np.unique(ids), minlength=len(vocab_to_id))
    else:
        vector = np.zeros(len(vocab_to_id)) # vector length = 2000
    return ' '.join([str(x) for x in vector])

def preprocess_tfidf(text, vocab_to_id):
    ids = [vocab_to_id.get(x) for x in tokens(text) if vocab_to_id.get(x)]
    if ids:
        vectorizer = TfidfVectorizer(stop_words='english', vocabulary=vocab_to_id, strip_accents='unicode')
        response = vectorizer.fit_transform(ids)
        n_response = response.toarray()
        row_sum = n_response.sum(axis=1)
        length = len(row_sum)
        n_result = n_response/row_sum.reshape(length,1)
        position_NaNs = np.isnan(n_result)
        n_result[position_NaNs] = 0
        vector = sparse.csr_matrix(n_result)
    else:
        vector = np.zeros(len(vocab_to_id))   
    return ' '.join([str(x) for x in vector])


def main(args):
    data = model.data.Dataset(args.input)
    with open(args.vocab, 'r') as f:
        vocab = [w.strip() for w in f.readlines()]
    vocab_to_id = dict(zip(vocab, range(len(vocab))))

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    labels = {}
    for collection in data.collections:
        output_path = os.path.join(args.output, '{}.csv'.format(collection))
        with open(output_path, 'w', newline='') as f:
            w = csv.writer(f, delimiter=',')
            for y, x in data.rows(collection, num_epochs=1):
                if y not in labels:
                    labels[y] = len(labels)
                w.writerow((labels[y], preprocess_tf(x, vocab_to_id)))
                # w.writerow((labels[y], preprocess_tfidf(x, vocab_to_id)))

    with open(os.path.join(args.output, 'labels.txt'), 'w') as f:
        f.write('\n'.join([k for k in sorted(labels, key=labels.get)]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='path to the input dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='path to the output dataset')
    parser.add_argument('--vocab', type=str, required=True,
                        help='path to the vocab')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
