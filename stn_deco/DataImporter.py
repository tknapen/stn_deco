#!/usr/bin/env python
# encoding: utf-8
"""
DataImporter.py

Created by Tomas Knapen on 04-03-2016.
Copyright (c) 2016 VU. All rights reserved.
"""

from __future__ import division

import os, sys, subprocess, shutil

import numpy as np
import scipy as sp

import pandas as pd

from IPython import embed as shell


FOLDER_NAMES = ["Test1", "Test2", "Test3"]
NUISANCES = ['WM', 'GM', 'CV']
COLLAPSED_ROIS = ['SThL', 'SThR', 'STRL', 'STRR']

class DataImporter(object):
	"""
	DataContainer is a class for objects that contain the data for this analysis.
	It abstracts the data from the input text files, and allows faster and easier access to the data in hdf5 format.
	"""
	def __init__(self, ssa):
		super(DataImporter, self).__init__()
		"""Empty constructor function"""	
		self.ssa = ssa

	#
	#
	#	First functionality is the import of event and roi data
	#
	#

	def import_fmri_file(self, 
						fmri_file, 
						skiprows = 4, 
						average = True, 
						filter = True, 
						window_length = 120, 
						polyorder = 3, 
						psc = True, 
						zscore_av = True):
		"""Import a single fmri file and return its contents.
		Savitzky Golay filtering is applied to the data on a per-voxel basis (filter = True), 
		window_length and polyorder are parameters applied to scipy.signal.savgol_filter.
		Window_length is in seconds, and is divided by TR. 
		Data can be averaged across voxels, with the 'average' argument. 
		Data can be z-scored over time, with the 'zscore_pv' and 'zscore_av' arguments, 
		for per-voxel and across-voxels zscoring. 
		"""
		data = np.loadtxt(fmri_file, skiprows = skiprows)
		# shell()
		if filter:
			wl = int(window_length / self.ssa.TR)
			if wl % 2 == 0:	# needs odd window_length
				wl += 1
			savgol_lp_data = sp.signal.savgol_filter(data, axis = 0, window_length = wl, polyorder = polyorder, mode = 'nearest')
			data = data - savgol_lp_data

		if psc:
			data = 100.0 * (data/(data + savgol_lp_data).mean(axis = 0))

		if average:
			data = data.mean(axis = 1)[:,np.newaxis]

		if zscore_av:
			data = (data-data.mean(axis = 0))/data.std(axis = 0)

		return data

	def import_fmri_data_files(self, fmri_files, filter = False, average_across_voxels = True, psc = False, zscore_av = True, skiprows = 4):
		"""Import all fmri files (from a given ROI), concatenate them and return them"""
		fmri_list_data = [self.import_fmri_file(f, filter = filter, average = average_across_voxels, psc = psc, zscore_av = zscore_av, skiprows = skiprows) for f in fmri_files]
		nr_trs_per_run = [fd.shape[0] for fd in fmri_list_data]
		self.ssa.logger.info([fd.shape for fd in fmri_list_data])
		# shell()
		return nr_trs_per_run, np.concatenate(fmri_list_data)

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
		nii_file_list = subprocess.Popen('ls ' + os.path.join(fmri_folder, FOLDER_NAMES[0], '*.txt'), 
			shell=True, stdout=subprocess.PIPE).communicate()[0].split('\n')[:-1]
		# the list of rois
		self.rois = [os.path.split(niif)[-1][:-len('.txt')].split('_')[-1] for niif in nii_file_list]
		# make sure event files aren't rois
		self.rois.remove('test')

	def import_moco_parameters_from_feats(self, base_folder_name = 'scFeat'):
		"""import_moco_parameters_from_feats takes moco parameters from a separate folder hierarchy.
		This method is post-hoc (after and separate from initial imports) 
		so it also saves the parameters in the hdf5 file.
		"""
		par_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

		moco_list_data = [np.loadtxt(os.path.join(self.ssa.base_dir.replace('Events', base_folder_name), self.ssa.subject_id, fn + '.feat', 'mc', 'prefiltered_func_data_mcf.par'))
			for fn in FOLDER_NAMES]

		moco_list_data = [(fd-fd.mean(axis = 0))/fd.std(axis = 0) for fd in moco_list_data]

		moco_pars = np.vstack(moco_list_data)
		moco_pars_dt = np.diff(np.vstack([np.zeros(moco_pars.shape[-1]), moco_pars]), axis = 0)
		moco_pars_ddt = np.diff(np.vstack([np.zeros(moco_pars_dt.shape[-1]), moco_pars_dt]), axis = 0)

		moco_pars_all = np.hstack((moco_pars, moco_pars_dt, moco_pars_ddt))

		par_all_names = np.vstack([par_names, [[p + '_' + d for p in par_names] for d in ['dt' ,'ddt']]])
		par_all_names = pd.Series(par_all_names.ravel())

		moco_pars_all_df = pd.DataFrame(moco_pars_all, columns = par_all_names )

		h5f = pd.HDFStore(self.ssa.hdf5_file, mode = 'a', complevel=9, complib='zlib' )
		# create original data group in the hdf5 file and put events in it
		h5f.put('original_data/moco_pars', moco_pars_all_df)
		h5f.close()


	def import_data(self):
		"""import_data imports all event and fmri data in a subject's folder. 
		"""
		# first, for the fMRI data
		self.find_rois(os.path.join(self.ssa.base_dir, self.ssa.subject_id))

		# dict to contain all imported fMRI data
		self.all_imported_roi_data = {}
		for roi in self.rois:
			self.ssa.logger.info('taking data from ' + roi)
			data_files = [os.path.join(self.ssa.base_dir, self.ssa.subject_id, fn, self.ssa.subject_id + '_MNI_' + roi + '.txt') 
							for fn in FOLDER_NAMES]
			if roi in NUISANCES: # these are single-voxel format
				self.nr_trs_per_run, rd = self.import_fmri_data_files(data_files, filter = True, average_across_voxels = False, psc = True, zscore_av = False, skiprows = 0)
			elif roi in COLLAPSED_ROIS: # these are single-voxel format
				# re-do the data_files names for the collapsed rois
				data_files = [os.path.join(self.ssa.base_dir, self.ssa.subject_id, fn, self.ssa.subject_id + '_MNI_weight_normalized_fsl2mm_' + roi + '.txt') 
					for fn in FOLDER_NAMES]
				self.nr_trs_per_run, rd = self.import_fmri_data_files(data_files, filter = True, average_across_voxels = False, psc = True, zscore_av = False, skiprows = 0)
			else: # these are multi-voxel format - all rois really
				self.nr_trs_per_run, rd = self.import_fmri_data_files(data_files, filter = True, average_across_voxels = False, psc = True, zscore_av = False, skiprows = 4)
			self.all_imported_roi_data.update({ roi: pd.DataFrame(rd) } )

		# then, we try to work out all the events
		self.event_files = [os.path.join(self.ssa.base_dir, self.ssa.subject_id, fn, self.ssa.subject_id + 'Event' + fn[-1] + '_test.txt') for fn in FOLDER_NAMES]
		self.all_event_data = self.import_event_files(self.event_files)

	def write_out_original_data(self):
		"""write out the imported data to the hdf5 file for this ssa.
		I'm using some compression for now, since it's a lot of data in all.
		"""
		try:
			os.system("rm " + self.ssa.hdf5_file)
		except OSError:
			pass
		h5f = pd.HDFStore(self.ssa.hdf5_file, mode = 'w', complevel=9, complib='zlib' )
		# create original data group in the hdf5 file and put events in it
		h5f.put('original_data/events', self.all_event_data)
		# save roi files
		for roi in self.rois:
			h5f.put('original_data/'+roi, self.all_imported_roi_data[roi])
		h5f.close()
