#!python
"""
Connects to Pivotal Tracker and downloads relevant app stories.

https://www.pivotaltracker.com/help/api/rest/v5#projects_project_id_activity
Exports

"""
import os
import pickle
import requests
import yaml

from utils import moveToSubfolder, pull_filenames, write_to_csv


with open("example.yaml", 'r') as stream:
    try:
        settings = dict(yaml.load(stream))
    except yaml.YAMLError as exc:
        print(exc, "check your settngs.yml for proper settings")

app_ids = settings.apps
token = settings.token

headers = {'X-TrackerToken': token}

def request():
    with requests.Session() as r:
        r.headers.update(headers)
        resp = r.get('https://www.pivotaltracker.com')

        for app, value in app_ids.items():
            current_iteration_number = \
                r.get('https://www.pivotaltracker.com/services/v5/projects/'
                      + str(value) + '?fields=current_iteration_number',
                      headers=headers).json() ['current_iteration_number'] - 1


            accepted_date = r.get('https://www.pivotaltracker.com/services/v5/projects/'
                      + str(value) + '/iterations?limit=1&offset'
                            '=' + str(current_iteration_number) + '&fields=start').json(
                                                                                )[0][
                                                                                'start']

            #You can use this date information to include in your final searches for both
            # the Current/Backlog versus Done panels, reformatted as MM/DD/YYYY
            accepted_date = accepted_date[5:7] + '/' + accepted_date[8:10] + '/' + \
                            accepted_date[:4]

            # All items in your Backlog/Current:
            current_stories = r.get(
                'https://www.pivotaltracker.com/services/v5/projects/' + str(value)
                + '/search?query=state:unstarted,started,finished,'
                                         'delivered,rejected OR accepted_since:' +
                accepted_date).json()
            current_stories = current_stories['stories']['stories']

            write_to_csv(app + '_current', current_stories)

            # For a panel that includes all Done stories:
            # check to see if .csv exists:
            files = pull_filenames(full_path=False)
            file_name = app + '_done.csv'
            if file_name in files:
                current = os.getcwd()
                os.chdir(current + '/last_pull')
                last_date = pickle.load(open(app + "_last_pull.p", "rb"))
                os.chdir(current)

                done_stories = r.get('https://www.pivotaltracker.com/services/v5/projects/'
                          + str(value) + '/search?query=accepted_since:'
                                     + last_date + '%20includedone:true').json()
                done_stories = done_stories['stories']['stories']
                write_to_csv(app + '_done', done_stories, append=True)

            else:
                # pull for first time:
                done_stories = r.get('https://www.pivotaltracker.com/services/v5/projects/'
                          + str(value) + '/search?query=accepted_before:'
                                     + accepted_date + '%20includedone:true').json()
                done_stories = done_stories['stories']['stories']
                write_to_csv(app + '_done', done_stories)

            # store date for later
            pickle.dump(accepted_date, open(app + '_last_pull.p', "wb"))
            moveToSubfolder('last_pull', app + '_last_pull.p')
