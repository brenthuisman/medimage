from . import __init__
'''
`image` factory for AVSField files. Can also save `image`s to AVSfield.
An AVSField is an XDR file with header (and may have a footer), roughly according to these specs:

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

Most of this file is based on Grey Hills work available at <https://github.com/greyhill/avsfld>.
'''

import ctypes
import numpy as np
import os
import xdrlib
import operator
import sys
from functools import reduce


def write_ndarray(path, v):
	'''
	works for any ndarray
	'''
	path = os.path.expanduser(path)
	fid = open(path, 'wb')

	lines = []
	# write the ascii header
	lines.append('# AVS field file (written by avsfld.py)\n')
	ndim = len(v.shape)
	lines.append('ndim=%d\n' % ndim)
	for d in range(ndim):
		lines.append('dim%d=%d\n' % (d + 1, v.shape[d]))
	lines.append('nspace=%d\n' % ndim)
	lines.append('veclen=1\n')

	if v.dtype == ctypes.c_float:
		lines.append('data=float_le\n')
	else:
		raise NotImplementedError('dtype %s not implemented' % str(v.dtype))
	lines.append('field=uniform\n')
	lines.append(chr(12)) #the magic two bytes
	lines.append(chr(12))

	lines = [l.encode('utf-8') for l in lines]

	fid.writelines(lines)
	fid.write(v.swapaxes(0, ndim - 1).tostring())
	fid.close()


def read(path):
	path = os.path.expanduser(path)
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
		p = os.path.dirname(path)
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
	elif header['data'] == 'float_le':
		# LE = little endian, no need to do order='f'
		if sys.byteorder != 'little':
			raise NotImplementedError('byte-swapping not implemented')
		raw_data = np.fromfile(fid, dtype=ctypes.c_float)
		# raw_data = np.fromfile(fid, dtype='<f4') # TODO: replace ctypes
		header['ElementType'] = 'MET_FLOAT_LE' # TODO UNKNOWN IF THIS WORKS
	elif header['data'] == 'byte':
				# bytesize is both LE and BE!
		raw_data = np.fromfile(fid, dtype='uint8')
		header['ElementType'] = 'MET_UCHAR'
	else:
		raise NotImplementedError(
			'datatype %s not implemented' % header['data'])

	fid.close()

	return __image_header(path,header), raw_data.reshape(tuple(reversed(shape))).swapaxes(0, ndim - 1)


def __image_header(path,xdrheader):
	newh = {}
	newh['ObjectType'] = 'Image'
	newh['ElementDataFile'] = os.path.basename(path.replace('.xdr','.raw')) #je moet toch wat
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

	# print(newh)

	# TODOs:
	# BinaryData = True
	# BinaryDataByteOrderMSB = False
	# CenterOfRotation = 0 0 0

	return newh