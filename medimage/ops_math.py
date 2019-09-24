import numpy as np,copy
from scipy import ndimage
import medimage as image

'''
Support mathematical operations. Take possibility of imdata being a masked array into account (using the filled() method mostly).

add/sub/mul/div methods invoke numpy methods, which may convert the type of your array to accomodate the result.
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

	def argmax(self):
		'''get index of highest value in image'''
		return np.unravel_index(np.nanargmax(self.imdata),self.imdata.shape)

	def argmin(self):
		'''get index of lowest value in image'''
		return np.unravel_index(np.nanargmin(self.imdata),self.imdata.shape)

	def add(self,other):
		if isinstance(other,image.image) and self.imdata.shape == other.imdata.shape:
			self.imdata = np.add(self.imdata,other.imdata)
		elif isinstance(other,float) or isinstance(other,int):
			self.imdata = np.add(self.imdata,other)
		else:
			raise TypeError("You're trying to add an unknown type or a differently sized image!")

	def sub(self,other):
		if isinstance(other,image.image) and self.imdata.shape == other.imdata.shape:
			self.imdata = np.subtract(self.imdata,other.imdata)
		elif isinstance(other,float) or isinstance(other,int):
			self.imdata = np.subtract(self.imdata,other)
		else:
			raise TypeError("You're trying to subtract an unknown type or a differently sized image!")

	def mul(self,other):
		if isinstance(other,image.image) and self.imdata.shape == other.imdata.shape:
			self.imdata = np.multiply(self.imdata,other.imdata)
		elif isinstance(other,float) or isinstance(other,int):
			self.imdata = np.multiply(self.imdata,other)
		else:
			raise TypeError("You're trying to multiply an unknown type or a differently sized image!")

	def div(self,other):
		with np.errstate(divide='ignore', invalid='ignore'):
			if isinstance(other,image.image) and self.imdata.shape == other.imdata.shape:
				self.imdata = np.true_divide(self.imdata,other.imdata)
			elif isinstance(other,float) or isinstance(other,int):
				self.imdata = np.true_divide(self.imdata,other)
			else:
				raise TypeError("You're trying to subtract an unknown type or a differently sized image!")

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

	def clip_range(self,mini,maxi):
		np.clip(self.imdata, mini, maxi,out=self.imdata)

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
		self.imdata = intercept+slope*self.imdata

	def density_to_materialindex(self,dens2mat_table):
		'''Convert this image from material densities to (continuous) materials indices, using the table you provide.'''
		materials = copy.deepcopy(dens2mat_table[1])
		dens2mat_table = copy.deepcopy(dens2mat_table)
		dens2mat_table[1]=list(range(len(dens2mat_table[0]))) #create material indices
		self.map_values(dens2mat_table)
		return materials # send to gpumcd

	def scale_values(self,rangemin,rangemax):
		''' Linear rescale of the pixelvalues to the provided interval. '''
		self.map_values([[self.imdata.min(),self.imdata.max()][rangemin,rangemax]])

	def map_values(self,table):
		'''Map the imdata-values of this image using the table you supply. This table should be a list of two equally long lists, where the first list maps to the current imdata-values, and the second to where you want them mapped. This function interpolates linearly, and does NOT extrapolate.'''
		assert len(table)==2
		assert len(table[0])==len(table[1])
		self.imdata = np.interp(self.imdata,table[0],table[1]) #type will be different!

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

	def compute_gamma_2(self,other,dta=3.,dd=3., local=False):
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
			'max_gamma': 15,
			'random_subset': None,
			'local_gamma': local,
			'ram_available': 2**30  # 1 GB
		}

		from pymedphys.gamma import gamma_shell, gamma_dicom

		retval.imdata = gamma_shell(tuple(self.get_axes_labels()), self.imdata, tuple(other.get_axes_labels()), other.imdata, **gamma_options)

		return retval

	def compute_gamma(self,other,dta=3.,dd=3., ddpercent=True,threshold=0.,defvalue=-1.,verbose=False):
		"""
		Compare two images with equal geometry, using the gamma index formalism as introduced by Daniel Low (1998).
		* ddpercent indicates "dose difference" scale as a relative value, in units percent (the dd value is this percentage of the max dose in the reference image)
		* ddabs indicates "dose difference" scale as an absolute value
		* dta indicates distance scale ("distance to agreement") in millimeter (e.g. 3mm)
		* threshold indicates minimum dose value (exclusive) for calculating gamma values: target voxels with dose<=threshold are skipped and get assigned gamma=defvalue.
		Returns an image with the same geometry as the target image.
		For all target voxels that have d>threshold, a gamma index value is given.
		For all other voxels the "defvalue" is given.
		If geometries of the input images are not equal, then a `ValueError` is raised.

		Adapted from gamma_index_3d_equal_geometry in https://github.com/OpenGATE/GateTools/blob/master/gatetools/gamma_index.py, which is governed by the Apache 2 license.
		"""

		def _reldiff2(dref,dtarget,ddref):
			"""
			Convenience function for implementation of the following functions.
			The arguments `dref` and `dtarget` maybe scalars or arrays.
			The calling code is responsible for avoiding division by zero (make sure that ddref>0).
			"""
			ddiff=dtarget-dref
			reldd2=(ddiff/ddref)**2
			return reldd2

		aref=self.imdata
		atarget=other.imdata
		if aref.shape != atarget.shape:
			raise ValueError("input images have different geometries ({} vs {} voxels)".format(aref.shape,atarget.shape))
		if not np.allclose(self.spacing(),other.spacing()):
			raise ValueError("input images have different geometries ({} vs {} spacing)".format(self.spacing(),other.spacing()))
		if not np.allclose(self.origin(),other.origin()):
			raise ValueError("input images have different geometries ({} vs {} origin)".format(self.origin(),other.origin()))
		if ddpercent:
			dd *= 0.01*np.max(aref)
		relspacing = np.array(self.spacing(),dtype=float)/dta
		inv_spacing = np.ones(3,dtype=float)/relspacing
		g00=np.ones(aref.shape,dtype=float)*-1
		mask=atarget>threshold
		g00[mask]=np.sqrt(_reldiff2(aref[mask],atarget[mask],dd))
		nx,ny,nz = atarget.shape
		ntot = nx*ny*nz
		nmask = np.sum(mask)
		ndone = 0
		if verbose:
			print("Both images have {} x {} x {} = {} voxels.".format(nx,ny,nz,ntot))
			print("{} target voxels have a dose > {}.".format(nmask,threshold))
		g2 = np.zeros((nx,ny,nz),dtype=float)
		for x in range(nx):
			for y in range(ny):
				for z in range(nz):
					if g00[x,y,z] < 0:
						continue
					igmax=np.round(g00[x,y,z]*inv_spacing).astype(int) # maybe we should use "floor" instead of "round"
					if (igmax==0).all():
						g2[x,y,z]=g00[x,y,z]**2
					else:
						ixmin = max(x-igmax[0],0)
						ixmax = min(x+igmax[0]+1,nx)
						iymin = max(y-igmax[1],0)
						iymax = min(y+igmax[1]+1,ny)
						izmin = max(z-igmax[2],0)
						izmax = min(z+igmax[2]+1,nz)
						ix,iy,iz = np.meshgrid(np.arange(ixmin,ixmax),
											np.arange(iymin,iymax),
											np.arange(izmin,izmax),indexing='ij')
						g2mesh = _reldiff2(aref[ix,iy,iz],atarget[x,y,z],dd)
						g2mesh += ((relspacing[0]*(ix-x)))**2
						g2mesh += ((relspacing[1]*(iy-y)))**2
						g2mesh += ((relspacing[2]*(iz-z)))**2
						g2[x,y,z] = np.min(g2mesh)
					ndone += 1
					if verbose and ((ndone % 1000) == 0):
						print("{0:.1f}% done...\r".format(ndone*100.0/nmask),end='')
		g=np.sqrt(g2)
		g[np.logical_not(mask)]=defvalue
		gimg=self.copy()
		gimg.imdata = g.astype(np.float32).copy()
		if verbose:
			print("100% done!     ")
		return gimg