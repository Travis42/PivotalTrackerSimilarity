#!python
"""
# Notice: word2vec...I didn't find it to be very useful.  Use tdidf.

Created by Travis Smith
travis42@gmail.com

Prototype to determine if it is possible to use NLP to make meaningful
comparisons between different projects within the Kessel Run app portfolio.

# useful keys:
# Title
# Description

dict_keys(['Id', 'Title', 'Labels', 'Iteration', 'Iteration Start',
'Iteration End', 'Type', 'Estimate', 'Current State', 'Created at',
'Accepted at', 'Deadline', 'Requested By', 'Description', 'URL',
'Owned By', 'Owned By.1', 'Owned By.2', 'Comment', 'Comment.1', 'Task',
'Task Status', 'Task.1', 'Task Status.1', 'Task.2', 'Task Status.2',
'Task.3', 'Task Status.3', 'Task.4', 'Task Status.4', 'Task.5',
'Task Status.5'])
"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import os
import string
import sys
import re
import multiprocessing
from pathlib import Path
import pickle
from time import time



def open_with_pandas_read_csv(df, filename):
    x = pd.read_csv(filename, sep=',', encoding='utf8')
    df[filename[:-4]] = x.loc[:, 'Title']
    df['temp'] = x.loc[:, 'Title']
    df.temp = df.temp.str.strip().str.lower()
    df.temp = df.temp.astype('str')
    df.temp = df.temp.dropna()
    df.temp = df.temp.str.encode('utf-8').str.decode('ascii', 'ignore')

    # getting rid of punctuation:
    df.temp = df.temp.apply(lambda x : re.sub('['+string.punctuation+']', '', x))

    stop = stopwords.words('english')
    stop.extend(['as', 'major', 'wing', 'i', 'want', 'so', 'that', 'should',
                 'be', 'able', 'acceptance', 'criteria', 'given', 'when',
                 'then',
                 'notes', 'because'])
    stop = set(stop)

    df.temp = df.temp.apply(lambda x: ' '.join(
                            [word for word in x.split() if word not in (stop)]))


    # tokenized, no stop words, result:
    df[filename[:-4] + ' tokenized'] = df.temp.apply(word_tokenize)
    df = df.drop('temp', axis=1)

    '''
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    '''
    #data = pd.DataFrame(data).to_dict().values()
    return df


def get_csvs():
    files = []
    # just pull files from this current directory
    for folderName, subfolders, filenames in os.walk(os.getcwd()):
        for filename in filenames:
            if filename.endswith('.csv'):
                files.append(filename)
        break

    df = pd.DataFrame()
    for file in files:
        result = open_with_pandas_read_csv(df, file)
    """
    # just getting 1 file:
    result = open_with_pandas_read_csv(files[0])
    """
    return result



def doc2vec_creation(tokens):
    """
    Creates a doc2vec model based on the inputs.
    :param tokens: a list of tokenized words
    :return: model: a doc2vec representation of the DataFrame
    """

    # MODEL PARAMETERS
    dm = 1  # 1 for distributed memory(default); 0 for dbow
    cores = multiprocessing.cpu_count()
    size = 300
    context_window = 50
    seed = 42
    min_count = 1
    alpha = 0.5
    max_iter = 200

    # BUILD MODEL
    model = Doc2Vec(documents=tokens,
                    dm=dm,
                    alpha=alpha,
                    # initial learning rate
                    seed=seed,
                    min_count=min_count,
                    # ignore words with freq less than min_count
                    max_vocab_size=None,  #
                    window=context_window,
                    # the number of words before and after to be used as context
                    size=size, # is the dimensionality of the feature vector
                    sample=1e-4,  # ?
                    negative=5,  # ?
                    workers=cores,  # number of cores
                    iter=max_iter
                    # number of iterations (epochs) over the corpus)
                    )
    return model

def doc2vec_creation_by_col(DataFrame, ColumnString):
    """
    Creates a doc2vec model based on the inputs.
    :param DataFrame: a Pandas DF containing tokenized words
    :param ColumnString: the name of a column to extract to the model
    :return: model: a doc2vec representation of the column
    """
    df = DataFrame
    column = ColumnString
    sentences = []
    for item_no, line in enumerate(df[column].values.tolist()):
        sentences.append(TaggedDocument(line, [item_no]))


    # MODEL PARAMETERS
    dm = 1  # 1 for distributed memory(default); 0 for dbow
    cores = multiprocessing.cpu_count()
    size = 300
    context_window = 50
    seed = 42
    min_count = 1
    alpha = 0.5
    max_iter = 200

    # BUILD MODEL
    model = Doc2Vec(documents=sentences, dm=dm, alpha=alpha,
                    # initial learning rate
                    seed=seed, min_count=min_count,
                    # ignore words with freq less than min_count
                    max_vocab_size=None,  #
                    window=context_window,
                    # the number of words before and after to be used as context
                    size=size, # is the dimensionality of the feature vector
                    sample=1e-4,  # ?
                    negative=5,  # ?
                    workers=cores,  # number of cores
                    iter=max_iter
                    # number of iterations (epochs) over the corpus)
                    )
    return model

def main():
    # round up all the data into a Pandas DataFrame, and prep it:
    df = get_csvs()

    # create a new DataFrame composed of all tokenized versions:
    tokenized = [col for col in df.columns if 'tokenized' in col]
    X = df.loc[:, tokenized]

    model = None


    # check to see if there is a model created, and load it if so:
    for folderName, subfolders, filenames in os.walk(os.getcwd()):
        # going with 1 month for now:
        path = Path('doc2vec.model')
        if 'doc2vec.model' in filenames and \
                time() - path.stat().st_mtime < (30 * 24 * 60 * 60): # 1 month
            # if model is past a certain date,destroy it and do the creation again
            model = Doc2Vec.load('doc2vec.model')
            sentences = pickle.load(open( "sentences.p", "rb" ))
        else:
            sentences = []
            for col in X.columns:
                for item_no, line in enumerate(X[col].values.tolist()):
                    sentences.append(TaggedDocument(line, [item_no]))
            model = doc2vec_creation(sentences)
            # 'slapshot_current_stories tokenized'
            # persist the model to disk:
            model.save('doc2vec.model')
            # freeze sentences as well:
            pickle.dump(sentences, open( "sentences.p", "wb" ))
        break


    # line 7 of Slapshot current stories
    tokens2 = "want when makes change Slapshot see only most " \
             "up-to-date information looking same day Slapshot " \
             "adds puck rink appear added".split()

    # example Triton discription item:
    tokens1 = "we need list mission type ATO USMTF know application " \
              "support".split()

    tokens  = "Major Wing should be able to drag the left most edge of any " \
             "puck to adjust its start time".split()


    new_vector = model.infer_vector(tokens)
    # gives you top 10 document tags and their cosine similarity
    sims = model.docvecs.most_similar([new_vector])
    print(sims)
    # top 3
    # so far, yes, it is getting 7 as the highest correlation.  That's good.
    results = sims[:5]
    for result in results:
        # check the score:
        if result[1] > 0.4:
            # TODO: use a different tokenizer from scratch (regex)
            print(sentences[result[0]])
            # df.loc[result[0], ['slapshot_current_stories']].tolist())
        else:
            print('no more matches found')
            break


if __name__ == '__main__':
    main()
