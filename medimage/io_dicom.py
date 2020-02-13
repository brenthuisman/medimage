'''
Loads dicom image into image. In case of a dicomseries directory, it contructs the 3D image manually as pydicom authors have stated not to include such functionality.

Optionally uses gdcm through SimpleITK.
'''
import glob,pydicom,numpy as np
from os import path
try:
	import SimpleITK as sitk
	SITK_PRESENT = True
except:
	SITK_PRESENT = False


def read(self,filename,**kwargs):
	'''
	TODO: account for PatientPosition (HFS and so on)
	'''
	self.header = {}
	dcm=None
	dcm_slices=None
	if path.isfile(filename):
		dcm = pydicom.dcmread(filename,force=True)
	elif path.isdir(filename):
		#probably 3D
		dcm_slices = [pydicom.dcmread(f,force=True) for f in glob.glob(path.join(filename,'*'))]
		dcm_slices = sorted(dcm_slices, key=lambda x: float(x.SliceLocation))
		dcm=dcm_slices[0]
	if SITK_PRESENT:
		read_sitk(self,filename)
	else: #non sitk path
		if path.isfile(filename):
			dcm = pydicom.dcmread(filename,force=True)
			self.imdata = dcm.pixel_array
			self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
			self.imdata = self.imdata.reshape(tuple(reversed(self.imdata.shape))).swapaxes(0, len(self.imdata.shape) - 1)
		elif path.isdir(filename):
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
			try:
				a=np.diff(dcm.GridFrameOffsetVector)
				b=np.unique(a)
				if len(b) == 1:
					self.header['ElementSpacing'].append(float(b))
			except:
				raise IOError("Nonlinear image opend: I don't handle this yet!")

		try:
			self.mul(dcm.DoseGridScaling) #doses may have this
		except:
			pass

		self.header['Offset'] = [float(i) for i in dcm.ImagePositionPatient]
		self.header['DimSize'] = list(self.imdata.shape)
		self.header['NDims'] = len(self.imdata.shape)

		## A few pieces of metadata that may be useful
		try:
			self.ct_to_hu(float(dcm.RescaleIntercept),float(dcm.RescaleSlope))
		except:
			print("This image appears not to be a CT, so I won't apply the rescaling that was not found!")
		#end of non sitk path
	try:
		self.PatientPosition = str(dcm.PatientPosition)
	except:
		pass
	try:
		self.DoseSummationType = str(dcm.DoseSummationType)
	except:
		pass


def read_sitk(self,filename):
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

	# so yeah, BOTH seem to be needed...
	self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
	self.imdata = self.imdata.reshape(tuple(reversed(self.imdata.shape))).swapaxes(0, len(self.imdata.shape) - 1)

	self.header['NDims'] = len(self.imdata.shape)
	self.header['DimSize'] = list(self.imdata.shape)


def write(self,filename):
	raise NotImplementedError("Can't write dicom files. See readme for explanation. Pull requests welcome!")
	#if SITK_PRESENT:
		#if str(self.imdata.dtype).startswith('float'):
			#print("The dicom file format does not support floating point data types.")
			#print("Multiplying all voxels by 100 and setting slope to 0.01 to preserve precision of a percentage.")
			#img=sitk.GetImageFromArray((self.imdata*100).astype(np.uint16))
			#img.SetMetaData("0028|1053","0.01") #slope
			#img.SetMetaData("0028|1052","0") #intercept
		#else:
			#img=sitk.GetImageFromArray(self.imdata)
		#img.SetSpacing(self.spacing())
		#img.SetOrigin(self.offset())
		#print("img.GetSpacing()",img.GetSpacing())
		#print("img.GetOrigin()",img.GetOrigin())
		#sitk.WriteImage(img,filename,False) #no compression
	#else:
		#raise ImportError("No SimpleITK present on this system, can't write dicom files...")
