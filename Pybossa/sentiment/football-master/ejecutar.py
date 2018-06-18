#!/usr/bin/env python2


from pybossa import tweetsForPybossaJson
from pybossa import tweetsForPybossaCsv
from pybossa import pybossaReport

## EXECUTE THE LINE BELOW BEFORE CREATING THE PROJECT
## IF YOU ARE GOING TO GIVE THE TWEETS IN A JSON FORMAT
#tweetsForPybossaJson('tweetsjson.json')

## EXECUTE THE LINE BELOW BEFORE CREATING THE PROJECT
## IF YOU ARE GOING TO GIVE THE TWEETS IN A CSV FORMAT
#tweetsForPybossaJson('tweetsjson.json')
tweetsForPybossaCsv()

## UNCOMMENT THE LINE BELOW WHEN THE USERS HAS ANSWERED
#pybossaReport()