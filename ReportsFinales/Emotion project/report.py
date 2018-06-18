#!/usr/bin/env python2

import sys

from pybossa import *


##FUNCIONA
if (sys.argv[1] == 'json'):
	jsonfile = sys.argv[2]
	if (len(sys.argv) > 3):
		goldencsvpath = sys.argv[3]
	else:
		goldencsvpath = None
	print(jsonfile)
	tweetsForPybossaJson(jsonfile, goldencsvpath)

##FUNCIONA
elif (sys.argv[1] == 'csv'):
	csvfile = sys.argv[2]
	if (len(sys.argv) > 3):
		goldencsvpath = sys.argv[3]
	else:
		goldencsvpath = None
	print(goldencsvpath)
	tweetsForPybossaCsv(csvfile, goldencsvpath)

##FUNCIONA
elif (sys.argv[1] == 'report'):
	filewithids = sys.argv[2]
	taskjsonpath = sys.argv[3]
	taskrunjsonpath = sys.argv[4]
	numberofcategories = int(sys.argv[5])
	if (len(sys.argv) > 6):
		goldencsvpath = sys.argv[6]
	else:
		goldencsvpath = None
	
	pybossaReport(filewithids, taskjsonpath, taskrunjsonpath, numberofcategories, goldencsvpath)


else:
	raise ValueError('Arguments not valid')