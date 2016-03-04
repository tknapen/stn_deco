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

from log import *

from DataImporter import *

class SSA(object):
	"""docstring for SSA"""
	def __init__(self, subject_id, TR, base_dir):
		super(SSA, self).__init__()
		for k,v in kwargs.items():
			setattr(self, k, v)

		# this ssa uses a specific hdf5 file for storing data
		self.hdf5_file = os.path.join(base_dir, subject_id + '.h5')

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

	def raw_roi_data(self, roi):
		h5f = pd.HDFStore(self.hdf5_file)
		raw_data = h5f['original_data/'+roi]
		h5f.close()
		return raw_data

	def rois(self):
		h5f = pd.HDFStore(self.hdf5_file)
		rois = [k.split('/')[-1] for k in h5f.keys() if ('original_data' in k) and (k != '/original_data/events')]
		h5f.close()
		return rois



	
		