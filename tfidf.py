#!python

from gensim import corpora, models, similarities
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import numpy as np

from API_interface import pull_filenames
from utils import moveToSubfolder

import os
import datetime
import string
import re


def open_with_pandas_read_csv(df, names, filename):
    x = pd.read_csv(filename, sep=',', encoding='utf8')
    # just get the filename, instead of full path:
    filename = filename.split("/")
    filename = filename[-1][:-4]

    df[filename] = x.loc[:, 'description']
    df['temp'] = x.loc[:, 'description']
    df.temp = df.temp.str.strip().str.lower()
    df.temp = df.temp.astype('str')
    df.temp = df.temp.dropna()
    df.temp = df.temp.str.encode('utf-8').str.decode('ascii', 'ignore')

    # getting rid of punctuation:
    df.temp = df.temp.apply(
        lambda x: re.sub('[' + string.punctuation + ']', '', x))

    stop = stopwords.words('english')
    stop.extend(
        ['as', 'major', 'wing', 'i', 'want', 'so', 'that', 'should', 'be',
         'able', 'acceptance', 'criteria', 'given', 'when', 'then', 'notes',
         'because', 'triton', 'slapshot', 'ellen', 'tad', 'chet', 'dana',
         'raven', 'ripsaw', 'chore', 'bug', 'threepio', 'spike', 'ian', 'fin',
         'dingo', 'jigsaw', 'mongo', 'eddie', 'skyhook', 'pythagoras',
         'marauder', 'rickshaw', 'tracer', 'blazon', 'tomás', 'tomas', 'mike',
         'avery', 'stacey', 'simone', 'home', 'one', 'oat', 'eeyore', 'aeron',
         'spacer', 'rebel', 'alliance', 'thaddeus', 'rickshaw', 'missy', 'tad',
         'chet', 'simone', 'dt', 'actor', 'chief', 'tracer', 'disco', 'archer',
         'dusty', 'persona'])
    stop = set(stop)

    df.temp = df.temp.apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    # tokenized, no stop words, result:
    df[filename[:-4] + ' tokenized'] = df.temp.apply(word_tokenize)
    df = df.drop('temp', axis=1)

    """
    # TODO: actually, using stemming leads to questionable results. Scores
    # are higher, but there's more trash.

    # TODO: try out some other stemmers to compare with.  This does seem to
    # get better results than nothing:
    # Stemmed by Lancaster Algo:
    lancaster_stemmer = LancasterStemmer()
    df[filename[:-4] + ' tokenized'] = \
        df[filename[:-4] + ' tokenized'].apply(lambda x: [
                                        lancaster_stemmer.stem(y) for y in x])
    """

    names[filename] = x.loc[:, 'name']
    return df, names


def get_csvs():
    files = pull_filenames()

    df = pd.DataFrame()
    names = pd.DataFrame()
    for file in files:
        df, names = open_with_pandas_read_csv(df, names, file)
    return df, names


def pull_story(file, first_app, first_title, second_app, second_title):
    """
    A greedy attempt to make a match between two app teams using the highest
     match % in each story match.  Its probably correct most of the time.
    """
    f = file
    first_app = first_app.split('_')[0]
    second_app = second_app.split('_')[0]
    f.write(first_app + ", your story: ")
    f.write('\n')
    f.write("'" + first_title + "'")
    f.write('\n')
    f.write("has some "
            "elements in common with " + second_app + "'s story called:")
    f.write('\n')
    f.write("'" + second_title + "'")
    f.write('\n')
    f.write('\n')
    f.write("Perhaps you should reach out to them and "
            "see if there are elements of this project you could work on "
            "together.")
    f.write('\n')


def write_user_story(f, title1, story1, chars):
    """
    Initial app's story
    :param f: an open file
    :param title1: the title of the story, string
    :param story1: the body of the story, string
    :param chars: number of characters desired in output, int
    """
    f.write('\n')
    f.write('\n')
    f.write('\n')
    f.write('User Story: ')
    f.write('\n')
    f.write(title1)
    f.write('\n')
    f.write('\n')
    f.write(story1[:chars])
    f.write('\n')
    f.write('\n')
    f.write('********')
    f.write('\n')
    f.write('Matches in Descending Order of '
            'Relevance:')
    f.write('\n')


def write_matching_story(file, appname, match_count, match, title, story,
                         chars):
    """
    Writes the story that matches the initial app's story.
    :param file: an open file
    :param appname: the name of the app, string
    :param match_count: a match #, int
    :param match: a percentage, float
    :param title: the story title, string
    :param story: the story body, string
    :param chars: amount of chars to show, int
    """
    f = file
    f.write('\n')
    f.write('______________')
    f.write('\n')
    f.write(appname + ':')
    f.write('\n')
    f.write('match #: ' + str(match_count))
    f.write('\n')
    f.write('match %: ' + str(round(match * 100)))
    f.write('\n')
    f.write(title)
    f.write('\n')
    f.write('\n')
    f.write(story[:chars])
    f.write('\n')
    f.write('\n')


def tfidf_process(sentences):
    """
    :param sentences: descriptions of stories
    :return: sims. Each position matches a position in sentences.
    # each value is a numpy ndarray that shows how much other sentences
    # match. The position of each of these corresponds to where they show up
    # in sentences.
    """

    ############### Processing Model
    # STEP 1 : Index and vectorize"

    # create dictionary (index of each element) (creates bag of words count)
    dictionary = corpora.Dictionary(sentences)
    # store the dictionary, for future reference
    # dictionary.save('sentences.dict')

    # compile corpus (vectors number of times each elements appears, bag of
    # words vectors)
    raw_corpus = [dictionary.doc2bow(t) for t in sentences]
    corpora.MmCorpus.serialize('sentences.mm', raw_corpus)  # store to disk

    # STEP 2 : Transform and compute similarity between corpuses"
    # load our dictionary
    dictionary = corpora.Dictionary.load('sentences.dict')

    # load vector corpus
    corpus = corpora.MmCorpus('sentences.mm')

    #########

    # Transform Text with TF-IDF
    tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model

    # convert our vectors corpus to TF-IDF space
    corpus_tfidf = tfidf[corpus]

    # STEP 3 : Create similarity matrix of all files
    index = similarities.MatrixSimilarity(tfidf[corpus])

    index.save('deerwester.index')
    index = similarities.MatrixSimilarity.load('deerwester.index')

    # get a similarity matrix for all documents in the corpus
    sims = index[corpus_tfidf]

    # print(list(enumerate(sims)))

    # print sorted (document number, similarity score) 2-tuples
    # print(sims[0])
    return sims


def tfidf():
    # round up all the data into a Pandas DataFrame, and prep it:
    df, names = get_csvs()

    # create a new DataFrame composed of all tokenized versions:
    tokenized = [col for col in df.columns if 'tokenized' in col]
    X = df.loc[:, tokenized]
    # drop from original so they're distinct:
    not_tokenized = [col for col in df.columns if 'tokenized' not in col]
    df = df.loc[:, not_tokenized]

    # transform to list of lists
    sentences = []
    for col in X.columns:
        for item_no, line in enumerate(X[col].values.tolist()):
            # create sentences to turn into model:
            sentences.append(line)

    # keywords have been extracted and stopwords removed.

    # for sims, each position matches a position in sentences.
    # each value is a numpy ndarray that shows how much other sentences
    # match. The position of each of these corresponds to where they show up
    # in sentences.
    sims = tfidf_process(sentences)

    # Create a map that references back to the original df:
    positions = {}
    last = None
    for col in df.columns:
        if last is None:
            for i in range(df[col].shape[0]):
                positions[i] = (col, df[col].loc[i], names[col].loc[i])
        else:
            # represents the largest position value from the last col:
            temp = max(positions.keys()) + 1
            for i in range(df[col].shape[0]):
                positions[temp + i] = (col, df[col].loc[i], names[col].loc[i])
        last = col

    results = {}
    for pos, value in enumerate(sims):
        # positions of highest values in array
        idx = np.argsort(value)
        temp = idx[-10:]  # top n (sorts least to greatest)
        topvals_index = temp[::-1]  # reversing to greatest first
        # actual metrics that are the top values
        top_matches = value[topvals_index]
        # exclude an exact match:
        if top_matches[0] >= 1.0:
            topvals_index = np.delete(topvals_index, [0])
            top_matches = value[topvals_index]
        elif top_matches[0] == 0.0:
            results[pos] = ['no match']
        else:
            pass

        # returns tokenized sentence
        # results[pos] = [sentences[i] for i in topvals_index

        # returns original sentence:
        results[pos] = {positions[index]: match for index, match in
                        zip(topvals_index, top_matches)}

    ###################

    # Give a read out of most relevant matches for a user app's current stories:

    apps = ['slapshot', 'triton', 'spacer', 'papercut', 'home_one',
            'rebel_alliance', 'raven', 'dingo', 'jigsaw', 'marauder', 'blazon',
            'ripsaw', 'chainsaw', 'disco']

    date = datetime.datetime.now().strftime("%m-%d-%Y")
    reports = {'current-done': True, 'current-current': False}
    for report, value in reports.items():
        with open(report + "_results_" + date + ".txt", "w") as f:
            hits = False
            for app in apps:
                f.write("Results for " + app)
                f.write('\n')
                for key, (appname, story1, title1) in positions.items():
                    # if no data, move on:
                    if type(appname[1]) == float:
                        continue
                    # select user's stories
                    current = app + '_current'
                    if appname == current:
                        user_story_written = False
                        match_count = 0
                        for (name, story2, title2), match in results[
                            key].items():
                            if app in name:
                                continue
                            elif type(story2) == float:
                                continue
                            elif match > 0.90:  # .95 is best
                                continue
                            elif match > 0.30:  # .25 or higher is best
                                # call out done items:
                                if ('done' in name) == value:
                                    match_count += 1
                                    hits = True
                                    if user_story_written == False:
                                        f.write('\n')
                                        f.write('--------------')
                                        f.write('\n')
                                        f.write('\n')
                                        pull_story(f, appname, title1, name,
                                                   title2)
                                        #write_user_story(f, title1, story1,
                                        # 200)
                                        user_story_written = True
                                    #write_matching_story(f, name, match_count,
                                                         #match, title2, story2,
                                                         #200)
                f.write('_______________________________________________')
                f.write('\n')
                f.write('\n')

        if hits is False:
            os.remove(report + "_results_" + date + ".txt")
        else:
            moveToSubfolder('results ' + str(date),
                            report + "_results_" + date + ".txt")
