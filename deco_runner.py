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
BASE_DIR = '/home/shared/STN_DECO/'


def run_subject(subject_id):

	ssa = SSA(subject_id = subject_id, base_dir = BASE_DIR, TR = TR)

	# preprocessing:
	# --------------
	# ssa.import_data()

	return True


#####################################################
#	parallel running of analyses in this function
#####################################################

def analyze_subjects(sjs, parallel = True ):
	if len(sjs) > 1 and parallel: 
		# parallel processing with joblib
		res = Parallel(n_jobs = len(sjs), verbose = 9)(delayed(run_subject)(sjs[i]) for i in range(len(sjs)))
	else:
		for i in range(len(sjs)):
			run_subject(sjs[i])
# 


#####################################################
#	main
#####################################################

def main():
	analyze_subjects([subjects[s] for s in subject_indices], \
		[run_arrays[s] for s in subject_indices], [session_dates[s] for s in subject_indices], \
		[projects[s] for s in subject_indices], \
		parallel = False)

if __name__ == '__main__':
	main()


