'''
This is an image library facilitating working with n dimensional images. The typical usecase is working with medical images, where you might want to slice, plot profiles, analyze, mask, or otherwise process medical image data. The imagedata is an ndarray, so it's very usable and extensible for all that are familiar with numpy.

The interal header info, keeping track of dimensions, which ndarrays don't do, is structured as in the MetaImage format, which you only need to know if you extend this library.

I started writing this lib because the Python bindings of ITK were difficult to install at the time (pre-simpleITK) and frankly the ITK API was and is very convoluted for the relatively simple things I wished and wish to do. Since I am very comfortable with the numpy library and the ndarray API, and the very simple data format of MetaImage I quickly could write a basic reader and writer, and from that the library sprawled to fit my needs. In my postdoc, I upgraded the library to Python 3, removed ROOT dependencies, and started a cleanup of the API, fixing a basic indexing issue that was always present and added AVSFIELD/XDR read/write support.
'''

import numpy as np,scipy
from os import path
from functools import reduce
from . import io_avsfield
from . import io_metaimage

#Functionality is split by moving function to another baseclass that we then inherit from
from .ops_math import math_class
from .ops_mask import mask_class

class image(math_class,mask_class):
    def __init__(self, infile, **kwargs):
        self.path,self.file = path.split(infile)

        for k,v in kwargs:
            setattr(self, k, v)

        # TODO copy "constructor"
        if infile.endswith('.mhd'):
            assert(path.isfile(infile))
            io_metaimage.read(self,infile)
        elif infile.endswith('.xdr'):
            assert(path.isfile(infile))
            io_avsfield.read(self,infile)
        else:
            print("Unrecognized file extension, aborting.")
            raise IOError()

        print(self.file,"loaded. Shape:",self.imdata.shape)


    def saveas(self,filename=None):
        if filename == None:
            raise FileNotFoundError("You must specify a filename when you want to save!")
        if len(filename.split(path.sep)) == 1: #so nothing to split, ie no dirs
            self.file = filename
        else:
            assert(path.isdir(filename.split()[0]))
            self.path,self.file = filename.split()
        fullpath = path.join(self.path,self.file)

        # VV doesnt support long, so we convert to int
        if self.imdata.dtype == np.int64:
            self.imdata = self.imdata.astype(np.int32, copy=False)
            self.datatype = '<i4'
            self.header['ElementType'] = 'MET_INT'
            print('MET_LONG not supported by many tools, so we autoconvert to MET_INT.')

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


    def get_pixel_index(self,coord,halfpixel=True):
        ''' get pixel at coord'''
        assert(len(coord)==3)

        x_x,x_y,x_z = self.get_axes_labels(halfpixel)

        return [
            min(range(len(x_x)), key=lambda i: abs(x_x[i]-coord[0])),
            min(range(len(x_y)), key=lambda i: abs(x_y[i]-coord[1])),
            min(range(len(x_z)), key=lambda i: abs(x_z[i]-coord[2])),
        ]


    def get_profiles_at_index(self,idx):
        ''' Returns the prependicular profiles through an index. '''
        assert len(idx)==len(self.imdata.shape)

        profiles = []

        for axi in range(len(self.imdata.shape)):
            idx_copy = idx[:]
            del idx_copy[axi]
            idx_copy = tuple(idx_copy)
            # print(idx_copy)
            profile = np.moveaxis(self.imdata,axi,-1)[idx_copy]
            # print(np.moveaxis(self.imdata,axi,-1).shape) #rollaxis didnt work correctly!
            # print(len(profile))
            profiles.append(profile)
        return profiles


    def get_line_atindex(self,axis,*args):
        ''' comment? '''
        print('*args',*args)
        if axis == 'x':
            return self.imdata[:,args[0],args[1]]
        if axis == 'y':
            return self.imdata[args[0],:,args[1]]
        if axis == 'z':
            return self.imdata[args[0],args[1],:]

