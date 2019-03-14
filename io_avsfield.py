'''
Read/write functionality for the AVS Field format. An AVSField is an XDR file with header (and may have a footer), roughly according to these specs:

source: http://vis.lbl.gov/archive/NERSC/Software/express/help6.2/help/reference/dvmac/Fiel0066.htm
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

AVSField header parsing is based on Grey Hills work available at <https://github.com/greyhill/avsfld>. Here, more complete header parsing and writing is added (extents), and writing to the (presumably) default BE format is implemented. xdrlib dependency is removed, as numpy reads and writes much faster.

Not supported:
* LE images (easily added, pull requests welcome)
* compressed images.
* unportable types (i.e. data=float means machine specific endianness)
* veclen =/= 1 (i.e. RGB per voxel) (easily added, pull requests welcome)
* nonuniform images

'''

import numpy as np
from os import path
import operator,sys
from functools import reduce

def write(self,infile):
	fid = open(infile, 'wb')

	lines = []
	# write the ascii header
	lines.append('# AVS field file (written by io_avsfield.py)\n')
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

	lines.append(chr(12)) #the magic two bytes
	lines.append(chr(12))

	lines = [l.encode('utf-8') for l in lines]

	fid.writelines(lines)
	# write as BE with correct target datatype
	dchar = self.imdata.dtype.char
	if dchar == "d":
		dchar = "f"
	if dchar == "h":
		dchar = "i"
	targettype = '>'+dchar+str(self.imdata.dtype.itemsize)
	self.imdata.swapaxes(0, ndim - 1).astype(targettype).tofile(fid)

	#write extents, looped pairwise over axis
	#xmin,xmax,ymin,ymax,zmin,zmax
	exts = np.ndarray(shape=2*len(minarr), dtype='>f4')
	for i,(mi,ma) in enumerate(zip(minarr,maxarr)):
		exts[2*i] = mi
		exts[2*i+1] = ma
	exts.tofile(fid)
	fid.close()
	print("New xdr file:",infile,file=sys.stderr)


def read(self,infile):
	''' currently extents are read from ascii header, not final bytes! those are skipped. '''
	fid = open(infile, 'rb')

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

	compression = 0
	try:
		compression = int(header['nki_compression'])
	except:
		pass #if not present or zero, then no compression.
	if compression is not 0:
		#raise NotImplementedError('This image has compressed imagedata, which is not supported by this library.')
		try:
			#must calc how many bytes of compressed data there are.
			image_data_offset = fid.tell()
			extent_data_offset = path.getsize(infile) - ndim * 2 * 4
			comp_size = extent_data_offset - image_data_offset

			import ctypes
			nki = ctypes.cdll.LoadLibrary(path.join(path.dirname(path.abspath(__file__)),'nki_decomp.dll'))

			# src = ctypes.c_char * comp_size
			# dest = ctypes.c_short * size
			# nki.nki_private_decompress(ctypes.byref(dest),ctypes.byref(src),len(src))

			src = np.fromfile(fid, dtype='uint8', count=comp_size)
			dest = np.zeros(size,dtype='<i2')

			nki.nki_private_decompress(dest.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),src.ctypes.data_as(ctypes.POINTER(ctypes.c_char)),len(src))

			# TODO: np.frombuffer ??

			raw_data = np.asarray(dest, order='F', dtype='<i2')

			# put fid at the extent
			assert(fid.tell() == extent_data_offset)#fid.seek(extent_data_offset)
		except:
			raise NotImplementedError('This image has compressed imagedata, but the external nki_decomp library cannot be found.')

	elif compression is 0:
		# external
		fname = None
		if 'variable 1 file' in list(header.keys()):
			fid.close()
			p=infile
			if p != '':
				p = '%s/' % p
			fname = '%s%s' % (p,header['variable 1 file'].split(' ')[0])
			fid = open(fname, 'rb')

		if header['data'] in ['xdr_real','xdr_float']:
			raw_data = np.asarray(np.fromfile(fid, dtype='>f4', count=size), order='F', dtype='<f4')
			# header['ElementType'] = 'MET_FLOAT'
		elif header['data'] == 'xdr_double':
			raw_data = np.asarray(np.fromfile(fid, dtype='>f8', count=size), order='F', dtype='<f8')
			# header['ElementType'] = 'MET_DOUBLE'
		elif header['data'] == 'xdr_short':
			raw_data = np.asarray(np.fromfile(fid, dtype='>i2', count=size), order='F', dtype='<i2')
			# header['ElementType'] = 'MET_SHORT'
		elif header['data'] == 'xdr_integer':
			raw_data = np.asarray(np.fromfile(fid, dtype='>i4', count=size), order='F', dtype='<i4')
			# header['ElementType'] = 'MET_INT'
		elif header['data'] == 'byte':
			# bytesize is both LE and BE!
			raw_data = np.fromfile(fid, dtype='uint8')
			# header['ElementType'] = 'MET_UCHAR'
		else:
			raise NotImplementedError(
				'datatype %s not implemented' % header['data'])

	#overwrite the header if it was ever present, AVSField standard mandates extents as final bytes, not header.
	header['min_ext'] = []
	header['max_ext'] = []

	for _ in range(ndim):
		header['min_ext'].append(np.fromfile(fid, dtype='>f4', count=1)[0])
		header['max_ext'].append(np.fromfile(fid, dtype='>f4', count=1)[0])

	assert ndim == len(header['min_ext']) == len(header['max_ext'])

	fid.close()

	self.header = __build_header(infile,header)
	self.imdata = raw_data.reshape(tuple(reversed(shape))).swapaxes(0, ndim - 1)


def __build_header(infile,xdrheader):
	newh = {}
	newh['ObjectType'] = 'Image'
	newh['ElementDataFile'] = '' #je moet toch wat
	newh['CompressedData'] = False
	newh['DimSize'] = xdrheader['DimSize']
	# newh['ElementType'] = xdrheader['ElementType']
	newh['NDims'] = int(xdrheader['ndim'])
	newh['TransformMatrix'] =  np.identity(newh['NDims']).flatten().tolist() # FIXME: read AVS transform from header or external field if present/provided.
	newh['Offset'] = [x*10. for x in xdrheader['min_ext']]
	newh['ElementSpacing'] = []
	for minn,maxx,nbin in zip(xdrheader['min_ext'],xdrheader['max_ext'],xdrheader['DimSize']):
		# nbin-1 because bincenter to bincenter, *10 to go to mm.
		newh['ElementSpacing'].append((maxx-minn)/float(nbin-1)*10.)

	# BinaryData = True
	# BinaryDataByteOrderMSB = False
	# CenterOfRotation = 0 0 0

	return newh