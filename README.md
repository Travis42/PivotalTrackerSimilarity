# PivotalTrackerSimilarity
Seeks to find simliar Stories among several Trackers in Pivotal Tracker

**Current Status: BROKEN.** Pivotal changed their API, and until I find a good use for this, I won't be learning their new one.
Project is on hold.

The implementation uses TFIDF to find similarities and presents those in a per-project .txt output.

To use:
- Put your tracker project name: id in the appropriate spot in settings.yml
- Put your token in the token variable, same file.
At the command line: python3 main.py

Future goals:  abstract several models into a class (model.py)
