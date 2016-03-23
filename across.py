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
# BASE_DIR = '/home/shared/STN_DECO/Events/'
BASE_DIR = '/home/shared/STN_DECO/MNIEvents/'

# check out the folders for all the different subjects
subject_folders = subprocess.Popen('ls -1 ' + BASE_DIR, 
			shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[:-1]

def run_across(subject_ids, base_dir, tr):
	# Aggregator works on a list of SSA objects.
	ssas = [SSA(subject_id = s, base_dir = base_dir, TR = tr) for s in subject_ids]

	# and this list is the only constructor argument
	agg = Aggregator(ssas)

	for roi in ['preSMAsmall', 'FFA23', 'maxSTN25exc', 'V1', 'PstriatumNoVentri', 'PvmPFCNoventri']:
	# for roi in ['FFA23', 'maxSTN25exc', 'V1']:
	# for roi in ['maxSTN25exc']:
		for et in ['ll']: # 'ww', 'wl_u', 
			# important categories:
			# 'SSRT', 'RT.ll', 'RT.ww', 'RT.wl_u'
			# agg.roi_deco_corrs(roi = roi, corr = ['alphaL','alphaG', 'SSRT', 'RTdiff.ll', 'RTdiff.ww'], event_type = et, name_suffix = 'all')
			# agg.roi_deco_corrs(roi = roi, corr = ['SSRT', 'alphaL','alphaG'], event_type = et, name_suffix = 'all')
			agg.roi_deco_corrs(roi = roi, corr = ['SSRT', "Acdiff.ll", 'RTdiff.ll'], event_type = et, name_suffix = 'SAR')
			agg.roi_deco_corrs(roi = roi, corr = ["Qdww", "Qdll", "Qdwl.u"], event_type = et, name_suffix = 'Q')
			# agg.roi_deco_corrs(roi = roi, corr = ['SSRT', 'medRTll', 'medRTww', 'medRTwl.u'], event_type = et, name_suffix = 'all')
			pass
		agg.roi_corrs(roi = roi, corr = ['SSRT', 'alphaL', 'alphaG'])
		agg.roi_deco_groups(roi = roi)

	pl.show()

	return True


#####################################################
#	main
#####################################################

def main():
	run_across(subject_folders, base_dir = BASE_DIR, tr = TR)

if __name__ == '__main__':
	main()


