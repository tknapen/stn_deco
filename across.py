#!/usr/bin/env python
# encoding: utf-8
"""
deo_runner.py

Created by Tomas Knapen on 04-03-2016.
Copyright (c) 2016 VU. All rights reserved.
"""

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib.pylab as pl

import os, sys, subprocess, shutil

from joblib import Parallel, delayed

from stn_deco.SSA import SSA
from stn_deco.Aggregator import Aggregator

TR = 2.0
# BASE_DIR = '/home/raw_data/STN_DECO/Events/'
# BASE_DIR = '/Users/knapen/Desktop/'
BASE_DIR = '/home/shared/STN_DECO/Events/'

# check out the folders for all the different subjects
subject_folders = subprocess.Popen('ls -1 ' + BASE_DIR, 
			shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[:-1]

def run_across(subject_ids, base_dir, tr):

	ssas = [SSA(subject_id = s, base_dir = base_dir, TR = tr) for s in subject_ids]

	agg = Aggregator(ssas)
	# agg.roi_deco_results(roi = 'maxSTN25exc')
	# agg.roi_deco_results(roi = 'V1')

	agg.roi_deco_corrs(roi = 'maxSTN25exc')
	agg.roi_deco_corrs(roi = 'V1')
	# agg.roi_deco_corrs(roi = 'FFA23')
	# agg.roi_deco_corrs(roi = 'GPe30exc')
	
	# agg.roi_deco_corrs(roi = 'preSMAsmall', event_type = 'wl_u')

	pl.show()

	return True


#####################################################
#	main
#####################################################

def main():
	run_across(subject_folders, base_dir = BASE_DIR, tr = TR)

if __name__ == '__main__':
	main()


