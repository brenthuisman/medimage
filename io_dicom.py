'''
Uses gdcm through SimpleITK.

TODO: use pydicom. for dicomdirs: contruct the 3D image ourselves. Loop over files in dir and use SliceLocation.
'''
from os import path

def read(self,filename):
	import SimpleITK as sitk #such that we don't require the package.
	self.header = {}
	sitk_image = None
	#https://stackoverflow.com/questions/40483190/simpleitk-tif-to-numpy-array-and-back-to-tif-makes-file-size-bigger
	#https://simpleitk.github.io/SimpleITK-Notebooks/01_Image_Basics.html
	#https://github.com/usuyama/pydata-medical-image/blob/master/lung_nodule/scripts/preprocess.py
	if path.isfile(filename):
		sitk_image=sitk.ReadImage(filename)
		self.imdata = sitk.GetArrayFromImage(sitk_image)
	elif path.isdir(filename):
		series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filename)
		series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filename, series_IDs[0])
		series_reader = sitk.ImageSeriesReader()
		series_reader.SetFileNames(series_file_names)
		# series_reader.MetaDataDictionaryArrayUpdateOn()
		series_reader.LoadPrivateTagsOn()
		sitk_image = series_reader.Execute()
		self.imdata = sitk.GetArrayFromImage(sitk_image)
	self.header['ElementSpacing'] = list(sitk_image.GetSpacing())
	self.header['Offset'] = list(sitk_image.GetOrigin())
	self.header['NDims'] = sitk_image.GetDimension()
	self.header['DimSize'] = list(sitk_image.GetSize())

	# so yeah, BOTH seem to be needed...
	self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
	self.imdata = self.imdata.reshape(tuple(reversed(self.imdata.shape))).swapaxes(0, self.header['NDims'] - 1)

	# and no idea where this comes from...
	#self.imdata = self.imdata/1e3


def write(self,filename):
	raise NotImplementedError("Writing to dicom objects is currently not validated.")
	#import SimpleITK as sitk
	# sitk_im = sitk.GetImageFromArray(self.imdata)
	# print(sitk_im.GetPixelIDTypeAsString())
	# sitk_im.SetOrigin(self.header['Offset'])
	# sitk_im.SetSpacing(self.header['ElementSpacing'])
	# sitk.WriteImage(sitk_im,filename)

