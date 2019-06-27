import numpy as np,copy
from scipy import ndimage
import image

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

	def ct_to_hu(self,intercept,slope):
		'''Convert this image from CT numbers to Hounsdield units, using the intercept and slope you provide.'''
		self.imdata = -intercept+slope*self.imdata

	def density_to_materialindex(self,dens2mat_table):
		'''Convert this image from material densities to (continuous) materials indices, using the table you provide.'''
		materials = copy.deepcopy(dens2mat_table[1])
		dens2mat_table = copy.deepcopy(dens2mat_table)
		dens2mat_table[1]=list(range(len(dens2mat_table[0]))) #create material indices
		self.map_values(dens2mat_table)
		return materials # send to gpumcd

	def map_values(self,table):
		'''Map the imdata-values of this image using the table you supply. This table should be a list of two equally long list, where the first list maps to the current imdata-values, and the second to where you want them mapped. This function interpolates linearly, and does NOT extrapolate.'''
		assert len(table)==2
		assert len(table[0])==len(table[1])
		self.imdata= np.interp(self.imdata,table[0],table[1]) #type will be different!

	def resample(self, new_ElementSpacing=[2,2,2], allowcrop=True, order=1):
		'''
		Resample image. Provide the desired ElementSpacing to which will be interpolated. Note that the spacing will be adjusted to obtain an integer image grid. Set allowcrop to True if you want to fix your new spacing and prefer to crop (subpixel distances) around the edges if necesary.
		'''

		old_ElementSpacing = np.array(self.header['ElementSpacing'])
		new_ElementSpacing = np.array(new_ElementSpacing)
		new_req_shape = self.imdata.shape * old_ElementSpacing / new_ElementSpacing
		new_shape = np.round(new_req_shape)

		if not allowcrop:
			#We keep the extent of the image fixed, so we'll adjust the new_ElementSpacing such that it fits into the new shape.
			real_resize_factor = new_shape / self.imdata.shape
			new_ElementSpacing = old_ElementSpacing / real_resize_factor

		new_shape = tuple(int(i) for i in new_shape) #new image expects tuples
		self.crop_as(image.image(ElementSpacing=new_ElementSpacing,Offset=self.header['Offset'],DimSize=new_shape))

		for s in self.imdata.shape:
			if s < 1:
				raise Exception('invalid image shape {}'.format(self.imdata.shape))

	def crop_as(self,other,**kwargs):
		'''
		Calculates the voxelvalues of this image on the grid of the provided image (other).

		Usage: create a new image, or use and existing one, with the desired grid, then supply that image to this function.

		If the other grid lays (partially) outside of self, then the area outside is extended as per `scipy.ndimage.map_coordinates` defaults.
		'''
		assert type(other)==type(self)

		# lets create indices for our new image
		indices = np.indices(other.imdata.shape)

		#now, we must transform these "new self" array indices (which we define as the indices of `other`) to array indices of "old self".
		#for this, we must go via the image indices, because those are in the same frame, unlike the image indices.
		#starting with the new (`other`) array indices, how to we get to the old (`self`) image indices? Convert other to world index, because those are the same, and then convert world index to self array index.

		for d in range(len(other.imdata.shape)):
			# from other array index to world coord
			worldcoord = indices[d]*other.header['ElementSpacing'][d]+other.header['Offset'][d]
			# from world coord to self array index.
			indices[d] = (worldcoord-self.header['Offset'][d])/self.header['ElementSpacing'][d]

		# Interpolation can fail (generate zero value voxels instead of interpolated values) for unsigned int types. Therefore, lets convert to double precision for the interpolation.
		oldtype=self.imdata.dtype
		self.imdata = ndimage.map_coordinates(self.imdata.astype(np.float64), indices, **kwargs).astype(oldtype)

		# correct metadata:
		self.header = copy.deepcopy(other.header)

	def compute_gamma(self,other,dta,dd, local=False):
		'''
		Requires pymedphys. Unfortunately, it's a _very_ slow calculation. Do not use unless you really have no other options.
		'''

		assert type(other)==type(self)
		retval = self.copy()

		gamma_options = {
			'dose_percent_threshold': dd,
			'distance_mm_threshold': dta,
			'lower_percent_dose_cutoff': 20,
			'interp_fraction': 5,  # Should be 10 or more for more accurate results
			'max_gamma': 2,
			'random_subset': None,
			'local_gamma': local,
			'ram_available': 2**30  # 1 GB
		}

		from pymedphys.gamma import gamma_shell, gamma_dicom

		retval.imdata = gamma_shell(tuple(self.get_axes_labels()), self.imdata, tuple(other.get_axes_labels()), other.imdata, **gamma_options)

		return retval
