#!python
"""
Created by Travis Smith

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

"""
I don't need to use icebox stories if I'm just comparing current and done.
It saves time, and further...it seems that results can change slightly 
depending on the overall corpus.

But...what could I do with those icebox stories?
Seems to be...not much.
"""

from API_interface import request
from tfidf import tfidf

from time import time

def main():
    t0 = time()
    request()
    tfidf()
    print("Done in %.3fs"%(time()-t0))

if __name__ == '__main__':
    main()