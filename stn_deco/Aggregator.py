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
import mne

import seaborn as sn
# sn.set(style="ticks")

from log import *
from IPython import embed as shell

#  function for plotting significance regions
def plot_significance_lines(data, time_points, offset, slope, p_value_cutoff = 0.05, pal = None):
	"""plot_significance_lines takes , or regions, or something, and calculates cluster-based p-values against 0.
		data numpy.array, subjects by time by conditions
		offset float, offset in y position for lines in plot
		slope float, change in y position for consecutive lines in plot
		p_value_cutoff float, below which p_value to plot.
	"""
	if pal is None:
		pal = sn.dark_palette("green", data.shape[-1])

	for i in range(data.shape[-1]): # loop across regions
		clusters = mne.stats.permutation_cluster_1samp_test(data[...,i])
		for cluster_times, p_val in zip (clusters[1], clusters[2]):
			if p_val < p_value_cutoff:
				s = np.arange(time_points.shape[0])[cluster_times]
				pl.plot([time_points[s[0]], time_points[s[-1]]], [offset + slope * i, offset + slope * i], c = pal[i], linewidth = 3.0, alpha = 0.8)


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

	def roi_deco_corrs(self, 
				roi = 'maxSTN25exc', 
				corr = ['SSRT', 'RTdiff.ll', 'RTdiff.ww', "acww", "acll", "acwl.u"], 
				event_type = 'wl_u',
				name_suffix = '',
				group = 'deco_results'):
		"""Take deco results from a list of ssa objects, 
		using their gather_deco_results method. Then, we correlate the 
		tiecourses with a corr argument variable per timepoint"""

		roi_data_panel = pd.Panel({s.subject_id: s.gather_deco_results(roi, group = group) for s in self.ssas})
		roi_data = np.array(roi_data_panel)
		conditions = np.array(list(roi_data_panel.axes[-1]))
		
		roi_data_diffs = np.squeeze(np.array([	roi_data[:,:,conditions=='ll'] - roi_data[:,:,conditions=='ww'], 
									roi_data[:,:,conditions=='ll'] - roi_data[:,:,conditions=='wl_u'], 
									roi_data[:,:,conditions=='ww'] - roi_data[:,:,conditions=='wl_u'] ]
									)).transpose(1,2,0)

		cond_diffs = ['ll-ww', 'll-wl_u', 'ww-wl_u']

		which_event_types = conditions != 'wl_l'

		event_types = self.ssas[0].gather_deco_results(roi).keys()
		times = pd.Series(self.ssas[0].gather_deco_results(roi).axes[0])

		et_index = np.arange(len(event_types))[event_types == event_type][0]

		for s in self.ssas:
			s.preprocess_events()

		par_dict = {par: np.array([np.array(s.evts[par])[0] for s in self.ssas]) for par in corr}

		f = pl.figure(figsize = (4,12))
		f.suptitle(roi)

		s = f.add_subplot(411)
		s.set_title('FIR responses')
		sn.tsplot(roi_data[:,:,which_event_types], time = times, condition = event_types[which_event_types], ci = 68)
		s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.set_xlabel('time [s]')
		s.set_ylabel('Z-scored BOLD')
		sn.despine(offset=10)

		s = f.add_subplot(412)
		s.set_title('FIR response differences')
		sn.tsplot(roi_data_diffs, time = times, condition = cond_diffs, ci = 68)
		s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.set_xlabel('time [s]')
		s.set_ylabel('Z-scored BOLD')
		sn.despine(offset=10)

		s = f.add_subplot(413)
		s.set_title('event type ' + event_type + '\ncorrelations')

		for i, k in enumerate(corr):
			pl.plot(times, 
				[sp.stats.pearsonr(x, par_dict[k])[0] for x in roi_data[:,:,et_index].T], label = k, c = ['k','c','m','r','g','y'][i])

		pl.legend()
		s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.set_xlabel('time [s]')
		s.set_ylabel('Pearson')
		sn.despine(offset=10)
		s.set_xlim([np.min(times), np.max(times)])

		s = f.add_subplot(414)
		s.set_title('event type ' + event_type + '\ncorrelations')

		for i, k in enumerate(corr):
			pl.plot(times, 
				[-np.log10(sp.stats.pearsonr(x, par_dict[k])[1]) for x in roi_data[:,:,et_index].T], label = k, c = ['k','c','m','r','g','y'][i])

		pl.legend()
		s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.axhline(2.5, color = 'k', alpha = 0.5, lw = 1.5, ls = '--')
		s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.set_xlabel('time [s]')
		s.set_ylabel('-Log10(p)')
		sn.despine(offset=10)
		s.set_xlim([np.min(times), np.max(times)])

		pl.tight_layout()
		pl.savefig(os.path.join(os.path.split(self.ssas[0].base_dir[:-1])[0], 'figs', roi + '_' + event_type + '_corr_%s.pdf'%name_suffix))

	def roi_corrs(self, 
				roi = 'maxSTN25exc', 
				corr = ['SSRT', 'RTdiff.ll', 'RTdiff.ww', "acww", "acll", "acwl.u"], 
				name_suffix = ''):
		"""Take deco results from a list of ssa objects, 
		using their gather_deco_results method. Then, we correlate the 
		tiecourses with a corr argument variable per timepoint"""

		roi_data_panel = pd.Panel({s.subject_id: s.gather_deco_results(roi) for s in self.ssas})
		roi_data = np.array(roi_data_panel)
		conditions = np.array(list(roi_data_panel.axes[-1]))
		
		times = pd.Series(self.ssas[0].gather_deco_results(roi).axes[0])

		for s in self.ssas:
			s.preprocess_events()

		par_dict = {par: np.array([np.array(s.evts[par])[0] for s in self.ssas]) for par in corr}

		f = pl.figure(figsize = (12,8))
		f.suptitle(roi)
		for i, evt in enumerate(conditions):
			idx = conditions == evt
			rd = np.squeeze(roi_data[:,:,idx])
			
			s = f.add_subplot(3,len(conditions),i+1)
			s.set_title(evt)
			s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
			s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
			sn.tsplot(rd, time = times, ci = 68, c = 'k')			
			s.set_ylabel('Z-scored BOLD')
			sn.despine(offset=10)
			s.set_xlim([np.min(times), np.max(times)])
			# s.set_ylim([np.min(rd), np.max(rd)])
			s.set_ylim([-0.25, 0.25])

			correlations, pvals = np.array([[sp.stats.pearsonr(rd[:,tp], par_dict[par]) for tp in np.arange(times.shape[0])] for par in corr]).transpose(2,0,1)
			# pvals = np.array([[-np.log10(sp.stats.pearsonr(rd[:,tp], par_dict[par])[1]) for tp in np.arange(times.shape[0])] for par in corr])
			s = f.add_subplot(3,len(conditions),len(conditions) + i+1)
			s.set_title(evt)
			s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
			s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
			for c, par in enumerate(corr):
				pl.plot(times, correlations[c], label = par, c = ['k','c','m','r','g','y'][c])
			s.set_ylabel('Pearson\'s r')
			sn.despine(offset=10)
			pl.legend()
			s.set_xlim([np.min(times), np.max(times)])
			s.set_ylim([-0.75, 0.75])

			s = f.add_subplot(3,len(conditions), 2*len(conditions) + i+1)
			s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
			s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
			for c, par in enumerate(corr):
				pl.plot(times, -np.log10(pvals[c]), label = par, c = ['k','c','m','r','g','y'][c])
			s.axhline(2.5, color = 'k', alpha = 0.5, lw = 1.5, ls = '--')
			s.set_xlabel('time [s]')
			s.set_ylabel('-10Log(p)')
			sn.despine(offset=10)
			pl.legend()
			s.set_ylim([-0.01, 5.5])
			s.set_xlim([np.min(times), np.max(times)])

		pl.tight_layout()

		pl.savefig(os.path.join(os.path.split(self.ssas[0].base_dir[:-1])[0], 'figs', roi + '_' + 'corr_%s.pdf'%name_suffix))

	def roi_deco_groups(self, 
				roi = 'maxSTN25exc', 
				event_types = ['ll', 'wl_u', 'wl_l', 'ww'],
				name_suffix = 'SSRT'):
		"""Take deco results from a list of ssa objects, 
		using their gather_deco_results method. Then, we correlate the 
		tiecourses with a corr argument variable per timepoint"""

		roi_data_panel = pd.Panel({s.subject_id: s.gather_deco_results(roi) for s in self.ssas})
		roi_data = np.array(roi_data_panel)
		conditions = np.array(list(roi_data_panel.axes[-1]))

		# event_types = self.ssas[0].gather_deco_results(roi).keys()
		times = pd.Series(self.ssas[0].gather_deco_results(roi).axes[0])

		for s in self.ssas:
			s.preprocess_events()

		SSRT_group = np.array([np.array(s.evts[u'SSRT_group'])[0] == 'SSRT_long' for s in self.ssas])

		f = pl.figure(figsize = (12,4))
		f.suptitle(roi)
		for i, evt in enumerate(event_types):
			s = f.add_subplot(1,len(event_types),i+1)
			idx = conditions == evt
			this_condition_data = (roi_data[SSRT_group,:,idx],roi_data[-SSRT_group,:,idx])

			s.set_title(evt)
			sn.tsplot(this_condition_data[0], time = times, condition = ['long'], ci = 68, color = 'g')
			sn.tsplot(this_condition_data[1], time = times, condition = ['short'], ci = 68, color = 'r')
			s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
			s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
			s.set_xlabel('time [s]')
			s.set_ylabel('Z-scored BOLD')
			sn.despine(offset=10)
			s.set_xlim([np.min(times), np.max(times)])

		pl.tight_layout()
		pl.savefig(os.path.join(os.path.split(self.ssas[0].base_dir[:-1])[0], 'figs', roi + '_group_%s.pdf'%name_suffix))

	def roi_deco_corrs_final_SSRT(self, 
				roi = 'maxSTN25exc', 
				event_type = 'll',
				name_suffix = '',
				group = 'deco_results',
				stats_threshold = 0.05):
		"""Take deco results from a list of ssa objects, 
		using their gather_deco_results method. Then, we correlate the 
		tiecourses with a corr argument variable per timepoint"""

		roi_data_panel = pd.Panel({s.subject_id: s.gather_deco_results(roi, group = group) for s in self.ssas})
		roi_data = np.array(roi_data_panel)
		roi_data_dt = np.hstack((np.zeros((roi_data.shape[0], 1, roi_data.shape[-1])), np.diff(roi_data, axis = 1)))
		conditions = np.array(list(roi_data_panel.axes[-1]))

		roi_data_diffs = np.squeeze(np.array([	roi_data[:,:,conditions=='ll'] - roi_data[:,:,conditions=='ww'], 
									roi_data[:,:,conditions=='ll'] - roi_data[:,:,conditions=='wl_u'], 
									roi_data[:,:,conditions=='ww'] - roi_data[:,:,conditions=='wl_u'] ]
									)).transpose(1,2,0)

		cond_diffs = ['ll-ww', 'll-wl_u', 'ww-wl_u']

		which_event_types = (conditions != 'wl_l') & (conditions != 'wl_u')

		event_types = self.ssas[0].gather_deco_results(roi).keys()
		times = pd.Series(self.ssas[0].gather_deco_results(roi).axes[0])

		et_index = np.arange(len(event_types))[event_types == event_type][0]

		for s in self.ssas:
			s.preprocess_events()

		par_dict = {'SSRT': np.array([np.array(s.evts['SSRT'])[0] for s in self.ssas])}
		which_subjects_valid_ssrt = par_dict['SSRT'] != 0

		ssrt_stats = np.array([[sp.stats.pearsonr(x[which_subjects_valid_ssrt], par_dict['SSRT'][which_subjects_valid_ssrt]) for x in roi_data[:,:,et].T] for et in range(len(event_types))])
		significant_corr_times = ssrt_stats[et_index,:,1] < stats_threshold
		peak_corr_time = ssrt_stats[et_index,:,1] == ssrt_stats[et_index,:,1].min()

		ssrt_stats_dt = np.array([[sp.stats.pearsonr(x[which_subjects_valid_ssrt], par_dict['SSRT'][which_subjects_valid_ssrt]) for x in roi_data_dt[:,:,et].T] for et in range(len(event_types))])


		ll_average_sign_times = roi_data[which_subjects_valid_ssrt][:,peak_corr_time,-1].squeeze()
		which_times_peak = (times > 3) & (times < 6)
		which_times_trough = (times > 7) & (times < 13)

		# peak_values = roi_data[which_subjects_valid_ssrt][:,significant_corr_times,et_index][:,0]
		# trough_values = roi_data[which_subjects_valid_ssrt][:,significant_corr_times,et_index][:,-1]
		peak_values = roi_data[which_subjects_valid_ssrt][:,which_times_peak,et_index].max(axis = 1)
		trough_values = roi_data[which_subjects_valid_ssrt][:,which_times_trough,et_index].min(axis = 1)

		value_diffs_ll = peak_values - trough_values

		SSRT_group = np.array([np.array(s.evts[u'SSRT_group'])[0] == 'SSRT_long' for s in self.ssas])[which_subjects_valid_ssrt]

		f = pl.figure(figsize = (8,8))
		s = f.add_subplot(221)
		s.set_title('FIR responses')
		s.axvspan(np.array(times[significant_corr_times])[0], np.array(times[significant_corr_times])[-1], color = 'r', alpha = 0.1)
		sn.tsplot(roi_data[:,:,which_event_types], time = times, condition = event_types[which_event_types], ci = 68, color = ['g','r'])
		s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		# s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.set_xlim([0,15])
		s.set_xticks([0,5,10,15])
		s.set_ylabel('% signal change STN')
		sn.despine(offset=10)

		pl.tight_layout()

		inset = f.add_subplot(222)
		g = sn.regplot(x=par_dict['SSRT'][which_subjects_valid_ssrt], y=ll_average_sign_times, color='r')
		print sp.stats.pearsonr(par_dict['SSRT'][which_subjects_valid_ssrt], ll_average_sign_times)

		pl.title('Correlation')
		inset.set_xlabel('SSRT')
		inset.set_ylabel('% signal change STN ll')
		inset.set_ylim([-0.2,0.2])
		inset.set_xlim([0,600])
		inset.set_yticks([-0.2, 0, 0.2])
		inset.set_xticks([0, 300, 600])
		inset.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		sn.despine(offset=10)

		# shell()

		s3 = f.add_subplot(223)	
		for i, evt in enumerate(np.arange(len(which_event_types))[which_event_types]): 
			which_event = event_types[evt]
			# pvals = np.array([[-np.log10(sp.stats.pearsonr(rd[:,tp], par_dict[par])[1]) for tp in np.arange(times.shape[0])] for par in corr])
			# s.set_title(evt)
			# print correlations, pvals
			# for c, par in enumerate(corr):
			pl.plot(times, ssrt_stats[evt,:,0], label = which_event, c = ['g','r'][i])
			if which_event == 'll':
				sig_lims = np.array(ssrt_stats[evt,significant_corr_times,1])[[0,-1]]
				sig_corrs = np.array(ssrt_stats[evt,significant_corr_times,0])[[0,-1]]
				print 'siglims %s, corrs = %s'%(str(sig_lims), str(sig_corrs))
				
				s3.axhline(np.mean(sig_corrs), color = 'k', alpha = 0.25, lw = 2.5, ls = '--')
			print evt, i
		s3.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		# s3.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s3.set_ylabel('Pearson\'s r SSRT')
		sn.despine(offset=10)
		s3.set_xlim([0,15])
		s3.set_xticks([0,5,10,15])
		s3.set_ylim([-0.25, 0.5])
		s3.set_yticks([-0.25, 0, 0.25, 0.5])

		s3.set_xlabel('time [s]')
		pl.legend()

		this_condition_data = (roi_data[SSRT_group,:,et_index],roi_data[-SSRT_group,:,et_index])
		s = f.add_subplot(224)
		# s.set_title(evt)
		sn.tsplot(this_condition_data[0], time = times, condition = ['long SSRT'], ci = 68, color = 'k')
		sn.tsplot(this_condition_data[1], time = times, condition = ['short SSRT'], ci = 68, color = 'gray', alpha = 0.5)
		s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
		# s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change STN ll')
		sn.despine(offset=10)
		s.set_xlim([0,15])
		s.set_xticks([0,5,10,15])

		-np.log10(np.array([sp.stats.ttest_ind(this_condition_data[0][:,i], this_condition_data[1][:,i])[1] for i in range(len(times))]))


		pl.tight_layout()
		pl.savefig(os.path.join(os.path.split(self.ssas[0].base_dir[:-1])[0], 'figs', roi + '_ll_corr_scatter_%s.pdf'%name_suffix))
		pl.show()
		shell()

