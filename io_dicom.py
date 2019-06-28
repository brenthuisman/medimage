'''
Loads dicom image into image. In case of a dicomseries directory, it contructs the 3D image manually as pydicom authors have stated not to include such functionality.

Optionally uses gdcm through SimpleITK.
'''
import glob,pydicom,numpy as np
from os import path

def read(self,filename,**kwargs):
	'''
	TODO: account for PatientPosition (HFS and so on)
	'''
	if 'sitk' in kwargs and kwargs['sitk']:
		read_sitk(self,filename)
		return

	self.header = {}
	dcm=None

	if path.isfile(filename):
		dcm = pydicom.dcmread(filename)
		self.imdata = dcm.pixel_array
		self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
		self.imdata = self.imdata.reshape(tuple(reversed(self.imdata.shape))).swapaxes(0, len(self.imdata.shape) - 1)

	elif path.isdir(filename):
		#probably 3D
		dcm_slices = [pydicom.dcmread(f) for f in glob.glob(path.join(filename,'*'))]
		dcm_slices = sorted(dcm_slices, key=lambda x: float(x.SliceLocation))
		dcm=dcm_slices[0]
		shape = (int(dcm.Rows), int(dcm.Columns), len(dcm_slices))
		# self.imdata = np.vstack((sl.pixel_array for sl in dcm_slices))
		self.imdata = np.zeros(shape, dtype=dcm.pixel_array.dtype)
		for i,sl in enumerate(dcm_slices):
			pa = sl.pixel_array
			pa = pa.reshape(pa.shape[::-1])
			pa = pa.reshape(tuple(reversed(pa.shape))).swapaxes(0, len(pa.shape) - 1)
			self.imdata[:,:,i] = pa

	try:
		self.header['ElementSpacing'] = [float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]), float(dcm.SliceThickness)]
		#[dcm.SliceThickness]+list(dcm.PixelSpacing[::-1])
	except:
		self.header['ElementSpacing'] = [float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1])]

	self.header['Offset'] = [float(i) for i in dcm.ImagePositionPatient]
	self.header['DimSize'] = list(self.imdata.shape)
	self.header['NDims'] = len(self.imdata.shape)


def read_sitk(self,filename):
	print("HOERA!!!!!")
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


def write(self,filename):
	raise NotImplementedError("Writing to dicom objects is currently not validated.")
