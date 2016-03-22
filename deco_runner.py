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

import os, sys, subprocess, shutil

from joblib import Parallel, delayed

from stn_deco.SSA import SSA

TR = 2.0
# BASE_DIR = '/home/raw_data/STN_DECO/Events/'
# BASE_DIR = '/Users/knapen/Desktop/'
# BASE_DIR = '/home/shared/STN_DECO/Events/'
BASE_DIR = '/home/shared/STN_DECO/MNIEvents/'

# check out the folders for all the different subjects
subject_folders = subprocess.Popen('ls -1 ' + BASE_DIR, 
			shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[:-1]

def run_subject(subject_id, base_dir, tr):

	ssa = SSA(subject_id = subject_id, base_dir = base_dir, TR = tr)

	# preprocessing:
	# --------------
	# ssa.import_data()
	# ssa.import_moco_parameters_from_feats()

	for roi in ['preSMAsmall', 'FFA23', 'maxSTN25exc', 'V1', 'PstriatumNoVentri', 'PvmPFCNoventri']:
		ssa.deconvolution_roi(roi = roi, deco_sample_frequency = 4.0, deconvolution_interval = [-7,20])

	return True


#####################################################
#	parallel running of analyses in this function
#####################################################

def analyze_subjects(sjs, parallel = True ):
	if len(sjs) > 1 and parallel: 
		# parallel processing with joblib
		res = Parallel(n_jobs = 24, verbose = 9)(delayed(run_subject)(sjs[i], BASE_DIR, TR) for i in range(len(sjs)))
	else:
		for i in range(len(sjs)):
			run_subject(sjs[i], BASE_DIR, TR)


#####################################################
#	main
#####################################################

def main():
	analyze_subjects(subject_folders, parallel = True)

if __name__ == '__main__':
	main()


