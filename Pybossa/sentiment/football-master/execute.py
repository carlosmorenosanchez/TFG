#!/usr/bin/env python2

import sys

from pybossa import tweetsForPybossa
from pybossa import pybossaReport

if (sys.argv[1] == 'json'):
	tweetsForPybossa('tweetsjson.json')

elif ((sys.argv[1] == 'report') and (len(sys.argv) > 2) ):
		numberofcategories = int(sys.argv[2])
		pybossaReport(numberofcategories)

else:
	raise ValueError('The only valid parameters are json or report + number of categories')
