'''
This is an image library facilitating working with n dimensional images. The typical usecase is working with medical images, where you might want to slice, plot profiles, analyze, mask, or otherwise process medical image data. The imagedata is an ndarray, so it's very usable and extensible for all that are familiar with numpy.

The interal header info, keeping track of dimensions, which ndarrays don't do, is structured as in the MetaImage format, which you only need to know if you extend this library.

I started writing this lib because the Python bindings of ITK were difficult to install at the time (pre-simpleITK) and frankly the ITK API was and is very convoluted for the relatively simple things I wished and wish to do. Since I am very comfortable with the numpy library and the ndarray API, and the very simple data format of MetaImage I quickly could write a basic reader and writer, and from that the library sprawled to fit my needs. In my postdoc, I upgraded the library to Python 3, removed ROOT dependencies, and started a cleanup of the API, fixing a basic indexing issue that was always present and added AVSFIELD/XDR read/write support.
'''

import numpy as np,copy,logging,sys
from os import path
from functools import reduce
from . import io_avsfield
from . import io_metaimage

#Functionality is split by moving function to another baseclass that we then inherit from
from .ops_math import math_class
from .ops_mask import mask_class

class image(math_class,mask_class):
	'''
	Image class. Instantiate with
	- a valid filename (xdr or mhd filetypes) to open an image.
	- 'DimSize' and 'ElementSpacing' in kwargs (optional: 'Offset') to create a new blank (=zeroed) image.
	'''

	def __init__(self, *args, **kwargs):
		if len(args) > 0 and path.isfile(args[0]):
			infile = str(args[0])
			self.path,self.file = path.split(infile)
			if infile.endswith('.mhd'):
				io_metaimage.read(self,infile)
			elif infile.endswith('.xdr'):
				io_avsfield.read(self,infile)
			else:
				raise IOError("Unrecognized file extension, aborting.")
			print(self.file,"loaded. Shape:",self.imdata.shape,file=sys.stderr)

		elif len(args) > 0 and not path.isfile(args[0]):
			raise IOError("Invalid filename provided: "+str(args[0]))

		elif len(args) == 0 and 'DimSize' in kwargs and 'ElementSpacing' in kwargs:
			#new blank image
			self.path,self.file=('','')
			self.header = {}
			self.header['ObjectType'] = 'Image'
			self.header['CompressedData'] = False
			self.header['DimSize'] = kwargs['DimSize']
			self.header['NDims'] = len(kwargs['DimSize'])
			self.header['ElementSpacing'] = kwargs['ElementSpacing']
			self.header['Offset'] = kwargs['Offset'] if 'Offset' in kwargs else [-x*((y-1)/2) for x,y in zip(self.header['ElementSpacing'],self.header['DimSize'])]

			dt='<f8'
			if 'dt' in kwargs:
				dt=kwargs['dt']
			self.imdata = np.zeros(self.header['DimSize'], dtype=dt)
			print("New image created. Shape:",self.imdata.shape,file=sys.stderr)

		else:
			raise IOError("Image instantiated with invalid parameters.")


	def copy(self):
		return copy.deepcopy(self)


	def saveas(self,filename=None):
		''' If you applied any masks, these voxels will be set to zero unless you set fillval. '''

		if filename == None:
			raise FileNotFoundError("You must specify a filename when you want to save!")
		if len(filename.split(path.sep)) == 1: #so nothing to split, ie no dirs
			self.file = filename
		else:
			assert(path.isdir(path.split(filename)[0]))
			self.path,self.file = path.split(filename)
		fullpath = path.join(self.path,self.file)

		# VV doesnt support long, so we convert to int
		if self.imdata.dtype == np.int64:
			self.imdata = self.imdata.astype(np.int32, copy=False)
			#self.datatype = '<i4'
			self.header['ElementType'] = 'MET_INT'
			print('MET_LONG not supported by many tools, so we autoconvert to MET_INT.',file=sys.stderr)

		if type(self.imdata) == np.ma.core.MaskedArray:
			print("Your masked array was squashed with the masked voxels set to",self.imdata.fill_value,file=sys.stderr)
			self.imdata = self.imdata.filled()

		# Update type, such that all writer write correct headers.
		if self.imdata.dtype.char+str(self.imdata.dtype.itemsize) == 'f4':
			self.header['ElementType'] = 'MET_FLOAT'
		if self.imdata.dtype.char+str(self.imdata.dtype.itemsize) == 'f8':
			self.header['ElementType'] = 'MET_DOUBLE'
		if self.imdata.dtype.char == 'd':
			# d (float64) is the default for most numpy generators, but is otherwise equiv to f8
			self.header['ElementType'] = 'MET_DOUBLE'
		if self.imdata.dtype.char+str(self.imdata.dtype.itemsize) == 'u1':
			self.header['ElementType'] = 'MET_UCHAR'
		if self.imdata.dtype.char+str(self.imdata.dtype.itemsize) == 'i2':
			self.header['ElementType'] = 'MET_SHORT'
		if self.imdata.dtype.char+str(self.imdata.dtype.itemsize) == 'h2':
			#sumtijms...
			self.header['ElementType'] = 'MET_SHORT'
		if self.imdata.dtype.char+str(self.imdata.dtype.itemsize) == 'i4':
			self.header['ElementType'] = 'MET_INT'
		if self.imdata.dtype.char+str(self.imdata.dtype.itemsize) == 'i8':
			self.header['ElementType'] = 'MET_LONG'

		# print(self.imdata.dtype)
		# print(self.header['ElementType'])

		if fullpath.endswith('.mhd'):
			io_metaimage.write(self,fullpath)
		elif self.file.endswith('.xdr'):
			io_avsfield.write(self,fullpath)

		return fullpath


	def get_axes_labels(self,halfpixel=False):
		''' Get axes labels. If halfpixel set to True, then it is assumed you want binedges, and you get axes labels for the edges (so N+1 labels for N bins)'''
		if halfpixel:
			hp=0.5
		else:
			hp=0
		axes = []
		for axi in range(len(self.imdata.shape)):
			axes.append( [float(self.header['Offset'][axi]+float(pos-hp)*self.header['ElementSpacing'][axi]) for pos in range(self.header['DimSize'][axi]+(1 if halfpixel else 0) )] )
		return axes


	def get_pixel_index(self,coord,halfpixel=False):
		''' get pixel at coord '''
		assert(len(coord)==len(self.imdata.shape))
		axl = self.get_axes_labels(halfpixel)
		return [min(range(len(axli)), key=lambda i: abs(axli[i]-coor)) for axli,coor in zip(axl,coord)]


	def get_profiles_at_index(self,idx):
		''' Returns the prependicular profiles through an index. '''
		assert len(idx)==len(self.imdata.shape)

		profiles = []

		for axi in range(len(self.imdata.shape)):
			idx_copy = idx[:]
			del idx_copy[axi]
			idx_copy = tuple(idx_copy)
			profile = np.moveaxis(self.imdata,axi,-1)[idx_copy]
			profiles.append(profile)
		return profiles


	def get_ctypes_pointer_to_data(self):
		import ctypes
		typecodes = np.ctypeslib._get_typecodes()
		ctypes_type = typecodes[self.imdata.__array_interface__['typestr']]
		return self.imdata.ctypes.data_as(ctypes.POINTER(ctypes_type))
