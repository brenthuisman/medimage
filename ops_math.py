import numpy as np
import sys

'''
ImdataIn these functions I try
'''

class math_class:
	def size(self):
		return self.imdata.size

	def nanfrac(self):
		try:
			return np.count_nonzero(self.imdata.filled(np.nan))/self.size()
		except:
			return np.count_nonzero(np.isnan(self.imdata))/self.size()

	def sum(self):
		return np.nansum(self.imdata)

	def max(self):
		return np.nanmax(self.imdata)

	def min(self):
		return np.nanmin(self.imdata)

	def std(self):
		try:
			return np.nanstd(self.imdata.filled(np.nan))
		except:
			return np.nanstd(self.imdata)

	def mean(self):
		try:
			return np.nanmean(self.imdata.filled(np.nan))
		except:
			return np.nanmean(self.imdata)

	def mean_excl_zero(self):
		return np.nanmean(self.imdata[self.imdata.nonzero()])

	def median(self):
		try:
			return np.nanmedian(self.imdata.filled(np.nan))
		except:
			return np.nanmedian(self.imdata)

	def percentiles(self,*percentiles):
		try:
			return np.nanpercentile(self.imdata.filled(np.nan),*percentiles)
		except:
			return np.nanpercentile(self.imdata,*percentiles)

	def smudge(self,mskval,frac=1.):
		#assume mskval must be ignored
		tmp = np.ma.masked_where(self.imdata == mskval, self.imdata)
		self.imdata[self.imdata != mskval] = tmp.mean()

	def tolowpass(self,threshold):
		self.imdata[self.imdata<threshold] = 0

	def tohighpass(self,threshold):
		self.imdata[self.imdata>threshold] = 0

	def normalize(self):
		self.imdata = self.imdata/self.imdata.max()

	def divide(self,N):
		with np.errstate(divide='ignore', invalid='ignore'):
			self.imdata = np.true_divide(self.imdata,N)

	def clip_range(self,mini,maxi):
		np.clip(self.imdata, mini, maxi,out=self.imdata)

	def argmax(self):
		'''get index of highest value in image'''
		return np.unravel_index(np.nanargmax(self.imdata),self.imdata.shape)

	def argmin(self):
		'''get index of lowest value in image'''
		return np.unravel_index(np.nanargmin(self.imdata),self.imdata.shape)

	def fill_gaussian_noise(self,mean,perc):
		assert 0. < perc < 100.
		self.imdata = np.random.normal(mean,mean*perc/100.,size=tuple(self.header['DimSize']))

	def passrate(self):
		'''Percentage of voxels with value <1. Usefull after gamma comp.'''
		return 100.*np.nansum(self.imdata < 1.)/self.imdata.count()
		# try:
		#     return 100.*np.nansum(self.imdata.filled(np.nan) < 1.)/np.count_nonzero(~np.isnan(self.imdata.filled(np.nan)))
		# except:
		#     return 100.*np.nansum(self.imdata < 1.)/np.count_nonzero(~np.isnan(self.imdata))

	def compute_gamma(self,other,dta,dd, local=False):
		assert type(other)==type(self)

		retval = self.copy()

		from npgamma import calc_gamma
		retval.imdata = calc_gamma(tuple(self.get_axes_labels()), self.imdata, tuple(other.get_axes_labels()), other.imdata, dta, self.max()*dd/100., 10, dta / 3, dta*2, np.inf, 16)

		# calc_gamma(
		#     coords_reference, dose_reference,
		#     coords_evaluation, dose_evaluation,
		#     distance_threshold, dose_threshold,
		#     lower_dose_cutoff=lower_dose_cutoff,
		#     distance_step_size=distance_step_size,
		#     maximum_test_distance=maximum_test_distance,
		#     max_concurrent_calc_points=max_concurrent_calc_points,
		#     num_threads=num_threads)

		# from pymedphys.gamma import gamma_shell
		# retval.imdata = gamma_shell(tuple(self.get_axes_labels()), self.imdata, tuple(other.get_axes_labels()), other.imdata, dd, dta, 10, dta, 10, local, None, True)

		# gamma_shell(coords_reference, dose_reference, coords_evaluation, dose_evaluation, dose_percent_threshold, distance_mm_threshold, lower_percent_dose_cutoff=20, interp_fraction=10, max_gamma=inf, local_gamma=False, global_normalisation=None, skip_when_passed=False)

		return retval

