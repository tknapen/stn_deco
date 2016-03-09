#!/usr/bin/env python
# encoding: utf-8
"""
Aggregator.py

Created by Tomas Knapen on 04-03-2016.
Copyright (c) 2016 VU. All rights reserved.
"""
from __future__ import division

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

import pandas as pd

import seaborn as sn
sn.set(style="ticks")

from log import *
from IPython import embed as shell


class Aggregator(object):
	"""docstring for Aggregator"""
	def __init__(self, ssas, **kwargs):
		super(Aggregator, self).__init__()
		for k,v in kwargs.items():
			setattr(self, k, v)

		self.ssas = ssas

		# setup logging for across subjects analysis.
		self.logger = logging.getLogger( self.__class__.__name__ )
		self.logger.setLevel(logging.DEBUG)
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)

	def roi_deco_results(self, roi = 'maxSTN25exc'):
		"""Take deco results from a list of ssa objects, 
		using their gather_deco_results method. Then, average them."""
		roi_data_panel = pd.Panel({s.subject_id: s.gather_deco_results(roi) for s in self.ssas})
		roi_data = np.array(roi_data_panel)

		event_types = self.ssas[0].gather_deco_results(roi).keys()
		times = pd.Series(self.ssas[0].gather_deco_results(roi).axes[0])

		f = pl.figure(figsize = (8,6))
		s = f.add_subplot(111)
		s.set_title('FIR responses')
		sn.tsplot(roi_data, time = times, condition = event_types)
		s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.set_xlabel('time [s]')
		s.set_ylabel('Z-scored BOLD')
		sn.despine(offset=10)

		pl.tight_layout()
		pl.savefig(os.path.join(os.path.split(self.ssas[0].base_dir[:-1])[0], 'figs', roi + '_deco.pdf'))

	def roi_deco_corrs(self, 
				roi = 'maxSTN25exc', 
				corr = ['alphaL','alphaG','SSRT'], 
				event_type = 'll'):
		"""Take deco results from a list of ssa objects, 
		using their gather_deco_results method. Then, we correlate the 
		tiecourses with a corr argument variable per timepoint"""

		roi_data_panel = pd.Panel({s.subject_id: s.gather_deco_results(roi) for s in self.ssas})
		roi_data = np.array(roi_data_panel)

		event_types = self.ssas[0].gather_deco_results(roi).keys()
		times = pd.Series(self.ssas[0].gather_deco_results(roi).axes[0])

		et_index = np.arange(len(event_types))[event_types == event_type][0]

		for s in self.ssas:
			s.preprocess_events()

		par_dict = {par: np.array([np.array(s.evts[par])[0] for s in self.ssas]) for par in corr}

		f = pl.figure(figsize = (4,8))

		s = f.add_subplot(311)
		s.set_title('FIR responses')
		sn.tsplot(roi_data, time = times, condition = event_types, ci = 95)
		s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.set_xlabel('time [s]')
		s.set_ylabel('Z-scored BOLD')
		sn.despine(offset=10)


		s = f.add_subplot(312)
		s.set_title(roi + '\nevent type ' + event_type + '\ncorrelations')

		for i, k in enumerate(par_dict.keys()):
			pl.plot(times, 
				[sp.stats.pearsonr(x, par_dict[k])[0] for x in roi_data[:,:,et_index].T], label = k, c = ['k','c','m'][i])

		pl.legend()
		s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.set_xlabel('time [s]')
		s.set_ylabel('Pearson')
		sn.despine(offset=10)
		s.set_xlim([np.min(times), np.max(times)])

		s = f.add_subplot(313)
		s.set_title(roi + '\nevent type ' + event_type + '\ncorrelations')

		for i, k in enumerate(par_dict.keys()):
			pl.plot(times, 
				[-np.log10(sp.stats.pearsonr(x, par_dict[k])[1]) for x in roi_data[:,:,et_index].T], label = k, c = ['k','c','m'][i])

		pl.legend()
		s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.axhline(2.5, color = 'k', alpha = 0.5, lw = 1.5, ls = '--')
		s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.set_xlabel('time [s]')
		s.set_ylabel('-Log10(p)')
		sn.despine(offset=10)
		s.set_xlim([np.min(times), np.max(times)])

		pl.tight_layout()
		pl.savefig(os.path.join(os.path.split(self.ssas[0].base_dir[:-1])[0], 'figs', roi + '_' + event_type + '_corr.pdf'))

