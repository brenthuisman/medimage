'''
Read/write functionality for the MetaImage format.

MetaIO is a library for reading and writing MetaImages, a file format used in OpenGate, ITK and associated toolkits. This library is an alternative implementation facilitating a bridge between numpy ndarrays and the MetaImage format.

More about the format at <https://itk.org/Wiki/ITK/MetaIO/Documentation#Quick_Start>.
'''

import sys
import numpy as np
from functools import reduce
from os import path

def read(self,filename):
    headerfile = open(filename,'r')
    self.header = {}
    datatype = None
    for line in headerfile:
        newline = line.strip()
        if len(newline)==0:
            continue
        newline=[x.strip() for x in newline.split('=')]

        try:
            self.header[newline[0]]=newline[1]
        except IndexError: #line without '='
            self.header[newline[0]]=None
        #at this point, header contains the full headerfile. now some prettyfication:

        if 'CompressedData' in newline[0] and 'True' in newline[1]:
            raise NotImplementedError("No valid input file (compressed).")
        if 'DimSize' in newline[0]:
            self.header['DimSize'] = [int(x) for x in newline[1].split()]
        if 'ElementType' in newline[0]:
            if 'MET_FLOAT' in newline[1]:
                datatype = '<f4'
            if 'MET_DOUBLE' in newline[1]:
                datatype = '<f8'
            if 'MET_UCHAR' in newline[1]:
                datatype = '<u1'
            if 'MET_SHORT' in newline[1]:
                datatype = '<i2'
            if 'MET_INT' in newline[1]:
                datatype = '<i4'
            if 'MET_LONG' in newline[1]:
                datatype = '<i8'
        if 'NDims' in newline[0]:
            self.header['NDims'] = int(newline[1])
        if 'TransformMatrix' in newline[0]:
            self.header['TransformMatrix'] = [float(x) for x in newline[1].split()]
        if 'CenterOfRotation' in newline[0]:
            self.header['CenterOfRotation'] = [float(x) for x in newline[1].split()]
        if 'Offset' in newline[0]:
            self.header['Offset'] = [float(x) for x in newline[1].split()]
        if 'ElementSpacing' in newline[0]:
            self.header['ElementSpacing'] = [float(x) for x in newline[1].split()]
    if 'LIST' in self.header['ElementDataFile']:
        print("We have a fake 4D file, assuming 3D...",file=sys.stderr)
        self.header['ElementDataFile'] = list(self.header.items())[-1][0]
        self.header['NDims'] -= 1
        self.header['DimSize'].pop()

    indata = np.asarray(np.fromfile(path.join(self.path,self.header['ElementDataFile']), dtype=datatype), order='F', dtype=datatype)

    if len(indata) == reduce(lambda x, y: x*y, self.header['DimSize']):
        self.nrvox = len(indata)
    else:
        raise IOError("The .mhd header info specified a different image size as was found in the .raw file.")

    self.imdata = indata.reshape(tuple(reversed(self.header['DimSize']))).swapaxes(0, self.header['NDims'] - 1)


def write(self,fullpath):
    self.header['ElementDataFile'] = self.file[:-4] + '.raw'
    fulloutraw = fullpath[:-4] + '.raw'
    #tofile is Row-major ('C' order), so that's why it happens to go correctly w.r.t. the HZYX order.
    self.imdata.swapaxes(0, self.header['NDims'] - 1).tofile(fulloutraw)
    print("New raw file:",fulloutraw,file=sys.stderr)
    with open(fullpath,'w+') as newheadf:
        newheadf.writelines("%s\n" % l for l in __getheaderasstring(self))
    print("New mhd file:",fullpath,file=sys.stderr)


def __getheaderasstring(self):
    #Convert self.header to string
    newhead = self.header.copy()
    for item in ['DimSize','NDims','TransformMatrix','CenterOfRotation','Offset','ElementSpacing']:
        #print self.header[item]
        try:
            newhead[item] = ' '.join(str(x) for x in self.header[item])
        except TypeError:
            newhead[item] = str(self.header[item])
        except KeyError:
            continue
    #order appears important for vv
    order = ['ObjectType','NDims','BinaryData','BinaryDataByteOrderMSB','CompressedData','TransformMatrix','Offset','CenterOfRotation','AnatomicalOrientation','ElementSpacing','DimSize','ElementType','ElementDataFile']
    newheadstr = []
    for item in order:
        if item in newhead:
            v = newhead[item]
            if v is None:
                newheadstr.append(str(item))
            else:
                newheadstr.append(str(item)+' = '+str(v))

    return newheadstr
