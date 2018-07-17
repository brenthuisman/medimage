import numpy as np
import os
from image import image
from decimal import Decimal
#from functools import reduce

#     WRITE_XDR: Successively is written to file:
           #- The string '#AVS wants ...'.
           #- The optional Header info
           #- The contents of the optional Header file
           #- An ascii description of the Input field
           #- Optionally the coordinates (coord%axis%[%pixel%]=%coord%)
             #(%axis% is 1 based, %pixel% 0 based, %coord% is float)
           #- Two bytes containing ascii character 0x0c
           #- The Data in binary (high byte first, unless native is selected).
             #Or, if NKI_Compression is greater than zero, compressed data.
           #- The Coordinates in binary IEEE float (high byte first, unless 
             #native is selected)

class xdrimage(image):
	def __init__(self, infile, **kwargs):
		#completely override init.

		#xdr consist of header part and bytes part. the border is 0x0c 0x0c
		border = None
		header = None
		# step 1: find border
		with open(filename, 'rb') as f:
			s = f.read()
			border = s.find(b'\x0c\x0c')
			header = s[0:border] #not sure this will work, because we read the file in binary
		
		# step 2: parse header
		#TODO
		
		# step 3: load image

		dtype = np.dtype([
			("time", np.float32),
			("PosX", np.float32),
			("PosY", np.float32),
			("Alt", np.float32),
			("Qx", np.float32),
			("Qy", np.float32),
			("Qz", np.float32),
			("Qw", np.float32),
			("dist", np.float32),
		])

		f = open("myfile", "rb")
		f.seek(border+2, os.SEEK_SET)

		data = np.fromfile(f, dtype=dtype)
		
		#dt = big endian. also, only real/float (>f4) and short/short (>i2) for now.
		self.imdata = np.fromfile(os.path.join(self.path,self.header['ElementDataFile']), dtype=dt)
