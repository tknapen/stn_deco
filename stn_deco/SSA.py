#!/usr/bin/env python
# encoding: utf-8
"""
SSA.py

Created by Tomas Knapen on 04-03-2016.
Copyright (c) 2016 VU. All rights reserved.
"""
from __future__ import division

import os
import tempfile, logging

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

import pandas as pd

import seaborn as sn
sn.set(style="ticks")
from fir import FIRDeconvolution

from log import *

from DataImporter import *

from IPython import embed as shell


class SSA(object):
	"""SSA is the main class for this analysis. 
	SSA stands for Single Subject Analysis. For each single-subject analysis a new object of this class is created. 

	"""
	def __init__(self, subject_id, TR, base_dir, **kwargs):
		"""SSA takes subject_id, TR and base_dir as arguments.
		subject_id is a string identifying the subject
		TR is the TR in seconds
		base_dir is the system path where the subjects' data folders are, and where the SSA's hdf5 file is stored. 
		
		"""
		super(SSA, self).__init__()
		for k,v in kwargs.items():
			setattr(self, k, v)

		self.subject_id = subject_id
		self.base_dir = base_dir
		self.TR = TR

		# this ssa uses a specific hdf5 file for storing data
		self.hdf5_file = os.path.join(base_dir, subject_id, subject_id + '.h5')

		# setup logging for this subject's analysis.
		self.logger = logging.getLogger( self.__class__.__name__ )
		self.logger.setLevel(logging.DEBUG)
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)

	def import_data(self):
		"""import_data uses the DataImporter class to take the txt file data
		to hdf5 format for future use."""
		dI = DataImporter(self)
		dI.import_data()
		dI.write_out_original_data()

	def events(self):
		h5f = pd.HDFStore(self.hdf5_file)
		raw_events = h5f['original_data/events']
		h5f.close()
		return raw_events

	def preprocess_events(self, event_conditions = ['ww', 'wl.u', 'wl.l', 'll']):
		self.evts = self.events()
		corr_incorr_trials = (self.evts['Cor_incor'] == 0 ) | ( self.evts['Cor_incor'] == 1 )
		stim_onset_times = self.evts['Time'] - self.evts['RT'] / 1000.0
		rts = self.evts['RT'] / 1000.0

		self.event_types_times = {evc: np.array(stim_onset_times[(self.evts['Cond'] == evc) & (corr_incorr_trials)]) for evc in event_conditions}
		self.event_types_durs = {evc: np.array(rts[(self.evts['Cond'] == evc) & (corr_incorr_trials)]) for evc in event_conditions}

	def raw_roi_data(self, roi):
		h5f = pd.HDFStore(self.hdf5_file)
		raw_data = h5f['original_data/'+roi]
		h5f.close()
		return raw_data

	def preprocess_roi_data(self, raw_data, pp_type = 'None'):
		if pp_type == 'Z':
			return (raw_data - raw_data.mean(axis = 0)) / raw_data.std(axis = 0)
		elif pp_type == 'psc':
			return 100 * (raw_data - raw_data.mean(axis = 0)) / raw_data.mean(axis = 0)
		elif pp_type == 'None':
			return raw_data

	def rois(self):
		h5f = pd.HDFStore(self.hdf5_file)
		rois = [k.split('/')[-1] for k in h5f.keys() if ('original_data' in k) and (k != '/original_data/events')]
		h5f.close()
		return rois

	def deconvolution_roi(self, 
					roi = 'maxSTN25exc', 
					event_types = ['ww', 'wl.u', 'wl.l', 'll'], 
					deco_sample_frequency = 4.0, 
					deconvolution_interval = [-7,18], 
					pp_type = 'Z',
					ridge = False):
		"""deconvolution_roi takes data from a ROI and performs deconvolution on it.
		arguments are roi (string), event_types (list of event type strings),
		deco_sample_frequency (float), interval (list of two timepoints)
		"""
		self.preprocess_events(event_conditions = event_types)
		data = self.preprocess_roi_data(self.raw_roi_data(roi), pp_type = pp_type)

		nuis_data = np.hstack([self.preprocess_roi_data(self.raw_roi_data(nuis_roi), pp_type = 'None') for nuis_roi in ['GM','WM','CV']])

		# first, we initialize the object
		fd = FIRDeconvolution(
						signal = data.squeeze(), 
						events = [np.array(self.event_types_times[evt]) for evt in event_types], 
						event_names = [evt.replace('.', '_') for evt in event_types], 
						durations = {evt.replace('.', '_'): np.array(self.event_types_durs[evt]) for evt in event_types},
						sample_frequency = 1.0/self.TR,
						deconvolution_frequency = deco_sample_frequency,
						deconvolution_interval = deconvolution_interval
						)

		# we then tell it to create its design matrix
		fd.create_design_matrix(intercept = True)

		nuis_data_resampled = sp.signal.resample(nuis_data.T, fd.resampled_signal_size, axis = -1)
		# fd.add_continuous_regressors_to_design_matrix(nuis_data_resampled)

		# perform the actual regression
		if ridge:
			fd.ridge_regress(cv = 20, alphas = np.r_[0,np.linspace(0.01, 100, 9)])	
		else:
			fd.regress(method = 'lstsq')

		# and partition the resulting betas according to the different event types
		fd.betas_for_events()

		fd.calculate_rsq()
		self.logger.info("%s Deco R squared is %1.3f" % (roi, fd.rsq[0]))

		f = pl.figure(figsize = (8,6))
		s = f.add_subplot(111)
		s.set_title('FIR responses, Rsq is %1.3f'%fd.rsq)
		for i, evt in enumerate([evt.replace('.', '_') for evt in event_types]):
			pl.plot(fd.deconvolution_interval_timepoints, fd.betas_for_cov(evt))

		pl.legend([evt.replace('.', '_') for evt in event_types], loc = 2)
		s.set_xlabel('time [s]')
		s.set_ylabel('Z-scored BOLD')
		sn.despine(offset=10)

		pl.tight_layout()
		pl.savefig(os.path.join(self.base_dir, self.subject_id, self.subject_id + '_' + roi + '.pdf'))

		deco_results = pd.DataFrame(np.squeeze([fd.betas_for_cov(evt.replace('.', '_')) for evt in event_types]).T, 
								index = fd.deconvolution_interval_timepoints, columns = [evt.replace('.', '_') for evt in event_types])

		h5f = pd.HDFStore(self.hdf5_file, mode = 'a', complevel=9, complib='zlib' )
		h5f.put('deco_results/%s'%(roi), deco_results)
		h5f.close()

	def gather_deco_results(self, roi):
		"""gather_deco_results takes pre-saved roi-based from the hdf5 file and returns them as the dataframe that they are.
		"""
		h5f = pd.HDFStore(self.hdf5_file, mode = 'r')
		deco_res = h5f.get('deco_results/%s'%(roi))
		h5f.close()

		return deco_res
	


						