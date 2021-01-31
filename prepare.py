# -*- coding: utf-8 -*-
# @Last Modified by:   Harshitha Rao
# @Last Modified time: 2020-12-05 03:27:45

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import random

def twenty_newsgroup_to_csv():
    newsgroups_train = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame( newsgroups_train.target_names)
    targets.columns=['title']

    out = pd.merge(df, targets, left_on='target', right_index=True)
    # out['date'] = pd.to_datetime('now')

    len_out = len(out)
    train_ix = random.sample(range(len_out), int(0.7*len_out))
    test_ix = random.sample(range(len_out), int(0.2*len_out))
    val_ix = random.sample(range(len_out), int(0.1*len_out))
    
    out = out[['title', 'text']]
    out.iloc[train_ix].to_csv('data/training.csv', index=False, header=False)
    out.iloc[test_ix].to_csv('data/test.csv', index=False, header=False)
    out.iloc[val_ix].to_csv('data/validation.csv', index=False, header=False)
    # out.to_csv('20_newsgroup.csv')
    
twenty_newsgroup_to_csv()