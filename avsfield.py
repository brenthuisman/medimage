'''
Read/write functionality for the AVS Field format. An AVSField is an XDR file with header (and may have a footer), roughly according to these specs:

WRITE_XDR: Successively is written to file:
	- The string '#AVS wants ...'.
	- The optional Header info
	- The contents of the optional Header file
	- An ascii description of the Input field
	- Optionally the coordinates (coord%axis%[%pixel%]=%coord%)
		(%axis% is 1 based, %pixel% 0 based, %coord% is float)
	- Two bytes containing ascii character 0x0c
	- The Data in binary (high byte first, unless native is selected).
		Or, if NKI_Compression is greater than zero, compressed data.
	- The Coordinates in binary IEEE float (high byte first, unless
		native is selected)
		- xmin,xmax,ymin,ymax,zmin,zmax (24 bytes), same as min max extents
		- deze extents zijn van pixelmiddens

Types:
	  { case AVS_TYPE_BYTE    : strcpy(temp, "data=byte\n"); break;
		case AVS_TYPE_SHORT   : strcpy(temp, "data=xdr_short\n"); break;
		case AVS_TYPE_INTEGER : strcpy(temp, "data=xdr_integer\n"); break;
		case AVS_TYPE_REAL    : strcpy(temp, "data=xdr_real\n"); break;
		case AVS_TYPE_DOUBLE  : strcpy(temp, "data=xdr_double\n"); break;
      { case AVS_TYPE_BYTE    : strcpy(temp, "data=le_byte\n"); break;
        case AVS_TYPE_SHORT   : strcpy(temp, "data=le_short\n"); break;
        case AVS_TYPE_INTEGER : strcpy(temp, "data=le_integer\n"); break;
        case AVS_TYPE_REAL    : strcpy(temp, "data=le_real\n"); break;
        case AVS_TYPE_DOUBLE  : strcpy(temp, "data=le_double\n"); break;

This file is based on Grey Hills work available at <https://github.com/greyhill/avsfld>. Here, more complete header parsing and writing is added, and writing to the (presumably) default BE format is implemented. NO LE supports is implemented
'''

import numpy as np
from os import path
import xdrlib
import operator
from functools import reduce

def write(self,path):
	fid = open(path, 'wb')

	lines = []
	# write the ascii header
	lines.append('# AVS field file (written by avsfield.py)\n')
	ndim = len(self.imdata.shape)
	lines.append('ndim=%d\n' % ndim)
	for d in range(ndim):
		lines.append('dim%d=%d\n' % (d + 1, self.imdata.shape[d]))
	lines.append('nspace=%d\n' % ndim)
	lines.append('veclen=1\n')

	if self.header['ElementType'] == 'MET_FLOAT':
		lines.append('data=xdr_real\n')
	elif self.header['ElementType'] == 'MET_DOUBLE':
		lines.append('data=xdr_double\n')
	elif self.header['ElementType'] == 'MET_SHORT':
		lines.append('data=xdr_short\n')
	elif self.header['ElementType'] == 'MET_INT':
		lines.append('data=xdr_integer\n')
	elif self.header['ElementType'] == 'MET_FLOAT':
		lines.append('data=xdr_real\n')
	elif self.header['ElementType'] == 'MET_UCHAR':
		lines.append('data=byte\n')
	else:
		raise NotImplementedError('dtype %s not implemented' % str(self.imdata.dtype))

	lines.append('field=uniform\n')

	minarr = [x*0.1 for x in self.header['Offset']] # mm to cm
	maxarr = [(o+(n-1)*bs)*0.1 for o,n,bs in zip(self.header['Offset'],self.imdata.shape,self.header['ElementSpacing'])]

	## VV appears not to use this header, but the final 24 bytes (for 3D images).
	lines.append('min_ext=')
	lines.append(' '.join(["%.2f"%i for i in minarr]))
	lines.append('\n')
	lines.append('max_ext=')
	lines.append(' '.join(["%.2f"%i for i in maxarr]))
	lines.append('\n')

	lines.append(chr(12)) #the magic two bytes
	lines.append(chr(12))

	lines = [l.encode('utf-8') for l in lines]

	fid.writelines(lines)
	# write as BE with correct target datatype
	targettype = '>'+self.imdata.dtype.char+str(self.imdata.dtype.itemsize)
	self.imdata.swapaxes(0, ndim - 1).astype(targettype).tofile(fid)

	#write extents, looped pairwise over axis
	#xmin,xmax,ymin,ymax,zmin,zmax
	exts = np.ndarray(shape=2*len(minarr), dtype='>f4')
	for i,(mi,ma) in enumerate(zip(minarr,maxarr)):
		exts[2*i] = mi
		exts[2*i+1] = ma
	exts.tofile(fid)
	fid.close()
	print("New xdr file:",path)


def read(self,path):
	fid = open(path, 'rb')

	# read the fld header
	ascii_header = []
	last_form_feed = False
	while True:
		next_char = fid.read(1)
		# end of ascii section
		if next_char == '' or (ord(next_char) == 12 and last_form_feed):
			break
		else:
			ascii_header.append(next_char.decode("utf-8"))
		last_form_feed = ord(next_char) == 12
	header_lines = (''.join(ascii_header)).split('\n')

	# parse fld header
	header = {}

	def parse_line(line):
		split = line.split('=')
		return (str(split[0]), ' '.join(split[1:]))

	for k, v in [parse_line(line) for line in header_lines]:
		header[k] = v

	# get interesting info from header
	ndim = int(header['ndim'])
	dimnames = ['dim%d' % (n + 1) for n in range(ndim)]
	shape = tuple([int(header[dimname]) for dimname in dimnames])
	header['DimSize'] = shape

	if header['field'] != 'uniform':
		raise NotImplementedError('field %s not implemented' % header['field'])

	size = reduce(operator.mul, shape)
	raw_data = None

	# external
	fname = None
	if 'variable 1 file' in list(header.keys()):
		fid.close()
		p=path
		if p != '':
			p = '%s/' % p
		fname = '%s%s' % (p,
						  header['variable 1 file'].split(' ')[0])
		fid = open(fname, 'rb')

	if header['data'] == 'xdr_real':  # same as xdr_float, but at NKI only real is used
		unpacker = xdrlib.Unpacker(fid.read())
		unpacked = unpacker.unpack_farray(size, unpacker.unpack_float)
		raw_data = np.asarray(unpacked, order='F', dtype='<f4')
		header['ElementType'] = 'MET_FLOAT'
	elif header['data'] == 'xdr_double':
		unpacker = xdrlib.Unpacker(fid.read())
		unpacked = unpacker.unpack_farray(size, unpacker.unpack_float)
		raw_data = np.asarray(unpacked, order='F', dtype='<f8')
		header['ElementType'] = 'MET_DOUBLE'
	elif header['data'] == 'xdr_short':
		unpacker = xdrlib.Unpacker(fid.read())
		unpacked = unpacker.unpack_farray(size, unpacker.unpack_float)
		raw_data = np.asarray(unpacked, order='F', dtype='<i2')
		header['ElementType'] = 'MET_SHORT'
	elif header['data'] == 'xdr_integer':
		unpacker = xdrlib.Unpacker(fid.read())
		unpacked = unpacker.unpack_farray(size, unpacker.unpack_float)
		raw_data = np.asarray(unpacked, order='F', dtype='<i4')
		header['ElementType'] = 'MET_INT'
	elif header['data'] == 'xdr_float':
		# slow path
		unpacker = xdrlib.Unpacker(fid.read())
		unpacked = unpacker.unpack_farray(size, unpacker.unpack_float)
		raw_data = np.asarray(unpacked, order='F', dtype='<f4')
		header['ElementType'] = 'MET_FLOAT'
	elif header['data'] == 'byte':
				# bytesize is both LE and BE!
		raw_data = np.fromfile(fid, dtype='uint8')
		header['ElementType'] = 'MET_UCHAR'
	else:
		raise NotImplementedError(
			'datatype %s not implemented' % header['data'])

	fid.close()

	self.header = __build_header(path,header)
	self.imdata = raw_data.reshape(tuple(reversed(shape))).swapaxes(0, ndim - 1)


def __build_header(path,xdrheader):
	newh = {}
	newh['ObjectType'] = 'Image'
	newh['ElementDataFile'] = '' #je moet toch wat
	newh['CompressedData'] = False
	newh['DimSize'] = xdrheader['DimSize']
	newh['ElementType'] = xdrheader['ElementType']
	newh['TransformMatrix'] = [float(x) for x in '1 0 0 0 1 0 0 0 1'.split()]
	newh['NDims'] = int(xdrheader['ndim'])
	newh['Offset'] = [float(x)*10. for x in xdrheader['min_ext'].split()]
	newh['ElementSpacing'] = []
	for minn,maxx,nbin in zip([float(x) for x in xdrheader['min_ext'].split()],[float(x) for x in xdrheader['max_ext'].split()],xdrheader['DimSize']):
		# nbin-1 because bincenter to bincenter, *10 to go to mm.
		newh['ElementSpacing'].append((maxx-minn)/float(nbin-1)*10.)

	# BinaryData = True
	# BinaryDataByteOrderMSB = False
	# CenterOfRotation = 0 0 0

	return newh