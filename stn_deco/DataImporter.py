#!/usr/bin/env python
# encoding: utf-8
"""
DataContainer.py

Created by Tomas Knapen on 04-03-2016.
Copyright (c) 2016 VU. All rights reserved.
"""

from __future__ import division

import os, sys, subprocess, shutil

import numpy as np
import scipy as sp

import pandas as pd

FOLDER_NAMES = ["Test1", "Test2", "Test3"]
NUISANCES = ['WM', 'GM', 'CV']

class DataImporter(object):
	"""
	DataContainer is a class for objects that contain the data for this analysis.
	It abstracts the data from the input text files, and allows faster and easier access to the data in hdf5 format.
	"""
	def __init__(self, ssa):
		super(DataContainer, self).__init__()
		"""Empty constructor function"""	
		self.ssa = ssa

	#
	#
	#	First functionality is the import of event and roi data
	#
	#

	def import_fmri_file(self, fmri_file, skiprows = 4):
		"""Import a single fmri file and return its contents"""
		return np.loadtxt(fmri_file, skiprows = skiprows)

	def import_fmri_data_files(self, fmri_files):
		"""Import all fmri files (from a given ROI), concatenate them and return them"""
		fmri_list_data = [self.import_fmri_file(f) for f in fmri_files]
		nr_trs_per_run = [fd.shape[0] for fd in fmri_list_data]
		return nr_trs_per_run, np.vstack(fmri_list_data)

	def import_event_file(self, event_file):
		"""Import a single event file and return its contents"""
		return pd.read_csv(event_file, sep = '\t')

	def import_event_files(self, event_files):
		"""Import all event files, concatenate them (and adding times) and return them"""
		assert hasattr(self, 'nr_trs_per_run')
		event_list_data = [self.import_event_file(ef) for ef in event_files]
		# shift the timings of the events for each of the runs, 
		# based on how many TRs there are in each of the runs
		shift_time = 0
		for ix, nr_trs in enumerate(self.nr_trs_per_run):
			event_list_data[ix]['Time'] += shift_time
			shift_time += nr_trs * self.ssa.TR
		all_evt_data = pd.concat(event_list_data)

		return all_evt_data

	def find_rois(self, fmri_folder = ''):
		"""Find the rois that have been saved in a specific folder."""
		nii_file_list = subprocess.Popen('ls ' + os.path.join(fmri_folder, '*.nii.gz.txt'), 
			shell=True, stdout=PIPE).communicate()[0].split('\n')[:-1]
		self.rois = [os.path.split(niif)[-1][:-len('.nii.gz.txt')] for niif in nii_file_list]		

	def import_data(self):
		"""import_data imports all event and fmri data in a subject's folder. 
		"""
		# first, we try to work out all the events
		self.event_files = [os.path.join(self.ssa.base_dir, fn, self.ssa.sj_index + 'Event' + fn[-1] + '_test.txt') for fn in FOLDER_NAMES]
		self.all_event_data = self.import_event_files(self.event_files)

		# now, for the fMRI data
		self.find_rois(os.path.join(self.ssa.base_dir, FOLDER_NAMES[0]))

		# dict to contain all imported fMRI data
		self.all_imported_roi_data = {}
		for roi in self.rois:
			rd = self.import_fmri_data_files([os.path.join(self.ssa.base_dir, fn, self.ssa.sj_index + '_' + roi + '.nii.gz.txt') 
										for fn in FOLDER_NAMES])
			if rd in NUISANCES: # average all nuisance voxels together for one timecourse: saves unnecessary space
				rd = rd.mean(axis = 1)[:,np.newaxis]
			self.all_imported_roi_data.update({ roi: pd.DataFrame(rd) } )

	def write_out_original_data(self):
		"""write out the imported data to the hdf5 file for this ssa"""
		h5f = pd.HDFStore(self.ssa.hdf5_file, mode = 'w' )
		# create original data group in the hdf5 file and put events in it
		h5f.put('original_data/events', self.all_event_data)
		# save roi files
		for roi in self.rois:
			h5f.put('original_data/'+roi, self.all_imported_roi_data[roi])
		h5f.close()
