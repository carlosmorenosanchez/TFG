#!/usr/bin/env python2

import sys

from pybossa import *

if ((sys.argv[1] == 'json') and (len(sys.argv) > 2)):
	jsonfile = sys.argv[2]
	print(jsonfile)
	tweetsForPybossa(jsonfile)

elif ((sys.argv[1] == 'csv') and (len(sys.argv) > 2)):
	csvfile = sys.argv[2]
	addGolden(csvfile)

elif ((sys.argv[1] == 'report') and (len(sys.argv) > 2) ):
		numberofcategories = int(sys.argv[2])
		pybossaReport(numberofcategories)

else:
	raise ValueError('The only valid parameters are json or report + number of categories')