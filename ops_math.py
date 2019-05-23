import numpy as np,scipy,copy

'''
Support mathematical operations. Take possibility of imdata being a masked array into account (using the filled() method mostly)
'''

class math_class:
	def size(self):
		return self.imdata.size

	def nanfrac(self):
		''' Include masked out bins as nan.'''
		return np.count_nonzero(np.isnan(np.ma.filled(self.imdata,fill_value=np.nan)))/self.imdata.size

	def sum(self):
		return np.nansum(np.ma.filled(self.imdata,fill_value=np.nan))

	def max(self):
		return np.nanmax(np.ma.filled(self.imdata,fill_value=np.nan))

	def min(self):
		return np.nanmin(np.ma.filled(self.imdata,fill_value=np.nan))

	def add(self,other):
		assert self.imdata.shape == other.imdata.shape
		self.imdata = self.imdata + other.imdata

	def std(self):
		return np.nanstd(np.ma.filled(self.imdata,fill_value=np.nan))

	def relunc(self):
		'''relative uncertainty (std / mean)'''
		return np.nanstd(np.ma.filled(self.imdata,fill_value=np.nan))/np.nanmean(np.ma.filled(self.imdata,fill_value=np.nan))

	def mean(self):
		return np.nanmean(np.ma.filled(self.imdata,fill_value=np.nan))

	def mean_excl_zero(self):
		return np.nanmean(self.imdata[self.imdata.nonzero()])

	def median(self):
		return np.nanmedian(self.imdata.filled(np.nan))

	def percentiles(self,*percentiles):
		return np.nanpercentile(self.imdata.filled(np.nan),*percentiles)

	def smudge(self,mskval,frac=1.):
		'''assume mskval must be ignored'''
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

	def hu_to_density(self,hu2dens_table):
		'''Convert this image from HU indices to materials densities, using the table you provide.'''
		self.map_values(hu2dens_table)
		self.imdata=self.imdata.astype('<f4')

	def ct_to_hu(self,intercept,slope):
		'''Convert this image from CT numbers to Hounsdield units, using the intercept and slope you provide.'''
		self.imdata = intercept+slope*self.imdata

	def density_to_materialindex(self,dens2mat_table):
		'''Convert this image from material densities to (continuous) materials indices, using the table you provide.'''
		materials = copy.deepcopy(dens2mat_table[1])
		dens2mat_table = copy.deepcopy(dens2mat_table)
		dens2mat_table[1]=list(range(len(dens2mat_table[0]))) #create material indices
		self.map_values(dens2mat_table)
		self.imdata=self.imdata.astype('<f4')
		return materials # send to gpumcd

	def map_values(self,table):
		'''Map the imdata-values of this image using the table you supply. This table should be a list of two equally long list, where the first list maps to the current imdata-values, and the second to where you want them mapped. This function interpolates linearly, and does NOT extrapolate.'''
		assert len(table)==2
		assert len(table[0])==len(table[1])
		self.imdata= np.interp(self.imdata,table[0],table[1]) #type will be different!

	def resample(self, new_ElementSpacing=[1,1,1],order=1):
		'''
		Resample image. Provide the desired ElementSpacing to which will be interpolated. Note that these will be rounded to obtain integer nbins.

		Spline interpolation can optionally be changed from bicubic.
		'''
		old_ElementSpacing = np.array(self.header['ElementSpacing'])
		new_ElementSpacing = np.array(new_ElementSpacing)
		new_real_shape = self.imdata.shape * old_ElementSpacing / new_ElementSpacing
		new_shape = np.round(new_real_shape)
		real_resize_factor = new_shape / self.imdata.shape
		new_ElementSpacing = old_ElementSpacing / real_resize_factor
		self.header['ElementSpacing'] = list(new_ElementSpacing)
		self.header['DimSize'] = list(new_shape)

		print('real_resize_factor',real_resize_factor)

		self.imdata = scipy.ndimage.zoom(self.imdata, real_resize_factor,order=order)

		if self.imdata.shape[0] < 1 or self.imdata.shape[1] < 1 or self.imdata.shape[2] < 1:
			raise Exception('invalid image shape {}'.format(self.imdata.shape))

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

		# from pymedphyData.gamma import gamma_shell
		# retval.imdata = gamma_shell(tuple(self.get_axes_labels()), self.imdata, tuple(other.get_axes_labels()), other.imdata, dd, dta, 10, dta, 10, local, None, True)

		# gamma_shell(coords_reference, dose_reference, coords_evaluation, dose_evaluation, dose_percent_threshold, distance_mm_threshold, lower_percent_dose_cutoff=20, interp_fraction=10, max_gamma=inf, local_gamma=False, global_normalisation=None, skip_when_passed=False)

		return retval

