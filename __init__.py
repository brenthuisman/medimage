'''
This is an image library facilitating working with n dimensional images. The typical usecase is working with medical images, where you might want to slice, plot profiles, analyze, mask, or otherwise process medical image data. The imagedata is an ndarray, so it's very usable and extensible for all that are familiar with numpy.

The interal header info, keeping track of dimensions, which ndarrays don't do, is structured as in the MetaImage format, which you only need to know if you extend this library.

I started writing this lib because the Python bindings of ITK were difficult to install at the time (pre-simpleITK) and frankly the ITK API was and is very convoluted for the relatively simple things I wished and wish to do. Since I am very comfortable with the numpy library and the ndarray API, and the very simple data format of MetaImage I quickly could write a basic reader and writer, and from that the library sprawled to fit my needs. In my postdoc, I upgraded the library to Python 3, removed ROOT dependencies, and started a cleanup of the API, fixing a basic indexing issue that was always present and added AVSFIELD/XDR read/write support.
'''

import numpy as np,pickle,subprocess,os
from decimal import Decimal
from functools import reduce
from . import avsfield
from . import metaimage

class image:
    def __init__(self, infile, **kwargs):
        self.path,self.file = os.path.split(infile)

        for k,v in kwargs:
            setattr(self, k, v)

        # TODO copy "constructor"
        if infile.endswith('.mhd'):
            assert(os.path.isfile(infile))
            metaimage.read(self,infile)
        elif infile.endswith('.xdr'):
            assert(os.path.isfile(infile))
            avsfield.read(self,infile)
        else:
            print("Unrecognized file extension, aborting.")
            raise IOError()

        print(self.file,"loaded. Shape:",self.imdata.shape)


    def __crush(self,crush):
        if self.crushed is True:
            print("Can't crush twice.")
            return
        if len(self.header['DimSize']) is 3 and len(crush) is 4:
            print("Assuming a valid 3D image is requested to be crushed, removing z-index...")
            crush.pop()
        if len(self.header['DimSize']) is not len(crush):
            print("Please supply proper dimensions.")
            return

        crush=crush[::-1]
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])

        ax = [i for (i,j) in zip(list(range(len(crush))),crush) if j==1]
        self.imdata = np.add.reduce(self.imdata, axis=tuple(ax))

        self.crushed = True

        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
        print("Image crushed. Shape:",self.imdata.shape)


    def __crush_argmax(self,crush):
        #crush, but doesnt project (=sum) but takes the max voxel-value)
        if self.crushed is True:
            print("Can't crush twice.")
            return
        # argmax does not support tuple axes so check that we only crush 1 dimension
        if not crush.count(1) == 1:
            print("Can't crush_argmax on more than 1 dimension.")
            return
        #if len(self.header['DimSize']) is 3 and len(crush) is 4:
            #print "Assuming a valid 3D image is requested to be crushed, removing z-index..."
            #crush.pop()
        if len(self.header['DimSize']) is not len(crush):
            print("Please supply proper dimensions.")
            return

        crush=crush[::-1]
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])

        ax = [i for (i,j) in zip(list(range(len(crush))),crush) if j==1]

        # argmax does not support tuple axes and casts to LONG.
        self.imdata = self.imdata.argmax(axis=ax[0])
        self.header['ElementType'] = 'MET_LONG'

        self.crushed = True
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
        print("Image crush_argmax'ed. Shape:",self.imdata.shape)


    def __crush_header(self,outpostfix,crush):
        self.header['NDims'] = crush.count(0)
        self.header['DimSize'] = [i for (i,j) in zip(self.header['DimSize'],crush) if j==0]
        self.header['Offset'] = [i for (i,j) in zip(self.header['Offset'],crush) if j==0]
        self.header['ElementSpacing'] = [i for (i,j) in zip(self.header['ElementSpacing'],crush) if j==0]
        self.header['ElementDataFile'] = self.file[:-4] + outpostfix + '.raw'

        try:
            self.header.pop('TransformMatrix') #FIXME: if ever needed.
            self.header.pop('CenterOfRotation')
        except KeyError:
            pass

        if self.header['NDims'] is 1:
            print("MHD doesnt support 1 dimensional images!")
            return


    def saveas(self,filename=None):
        if filename == None:
            raise FileNotFoundError("You must specify a filename when you want to save!")
        if len(filename.split(os.path.sep)) == 1: #so nothing to split, ie no dirs
            self.file = filename
        else:
            assert(os.path.isdir(filename.split()[0]))
            self.path,self.file = filename.split()
        fullpath = os.path.join(self.path,self.file)

        # VV doesnt support long, so we convert to int
        if self.imdata.dtype == np.int64:
            self.imdata = self.imdata.astype(np.int32, copy=False)
            self.datatype = '<i4'
            self.header['ElementType'] = 'MET_INT'
            print('MET_LONG not supported by many tools, so we autoconvert to MET_INT.')

        if fullpath.endswith('.mhd'):
            metaimage.write(self,fullpath)
        elif self.file.endswith('.xdr'):
            avsfield.write(self,fullpath)

        return fullpath


    def toprojection(self,outpostfix,crush):
        assert crush.count(0) is 2 or 3
        self.__crush(crush)
        self.__crush_header(outpostfix,crush)


    def save_crush_argmax(self,outpostfix,crush):
        assert crush.count(0) is 2 or 3
        self.__crush_argmax(crush)
        self.__crush_header(outpostfix,crush)
        return self.saveas(outpostfix)


    def saveprojection(self,outpostfix,crush):
        self.toprojection(outpostfix,crush)
        return self.saveas(outpostfix)


    def get1dlist(self,crush):
        #sums all the dimensions set to crush.
        assert crush.count(0) is 1
        #pgbins = self.imdata.shape[-1]

        crush=crush[::-1]
        outdata = self.imdata.copy()
        outdata = outdata.reshape(outdata.shape[::-1])
        ax = tuple([i for (i,j) in zip(list(range(len(crush))),crush) if j==1])
        out = np.add.reduce(outdata, axis=ax)

        return out


    def getprofile(self,axis):
        imdata = self.imdata.squeeze()
        assert len(imdata.shape) == 2
        if axis == 'x':
            spacing = float( self.header['ElementSpacing'][0] )
            print('Getting x profile at', int(imdata.shape[0]/2.))
            return ( np.linspace(0,spacing*imdata.shape[0],imdata.shape[0])/10. , imdata[int(imdata.shape[0]/2.)] )
        if axis == 'y':
            spacing = float( self.header['ElementSpacing'][1] )
            print('Getting y profile at', int(imdata.shape[1]/2.))
            return ( np.linspace(0,spacing*imdata.shape[1],imdata.shape[1])/10. , imdata.T[int(imdata.T.shape[1]/2.)] )


    def getline(self,axis):
        data = self.imdata.reshape(self.imdata.shape[::-1]) #same weird shit as getslice
        halfx=int(self.header['DimSize'][2]/2.)#swapped????, as getslice
        halfy=int(self.header['DimSize'][1]/2.)
        halfz=int(self.header['DimSize'][0]/2.)
        print(halfx,halfy,halfz)
        print(data.shape)
        if axis == 'x':
            #print((data[:,halfy,halfz]))
            return data[:,halfy,halfz]
            # return data[halfx,halfy,:] #swapped, as getslice
        if axis == 'y':
            #print((data[halfx,:,halfz]))
            return data[halfx,:,halfz]
        if axis == 'z':
            return data[halfx,halfy,:]


    def getline_atindex(self,axis,*args):
        ''' comment? '''
        print('*args',*args)
        if axis == 'x':
            return self.imdata[:,args[0],args[1]]
        if axis == 'y':
            return self.imdata[args[0],:,args[1]]
        if axis == 'z':
            return self.imdata[args[0],args[1],:]


    def get_axis_mms(self,axis,halfpixel=False):
        # op pixelmiddens

        # assen NIET omgekeerd...
        if halfpixel:
            hp=0.5
        else:
            hp=0.
        if axis == 'x':
            return [ float(self.header['Offset'][0]+(pos-hp)*self.header['ElementSpacing'][0]) for pos in range(self.header['DimSize'][0]) ]
        if axis == 'y':
            return [ float(self.header['Offset'][1]+(pos-hp)*self.header['ElementSpacing'][1]) for pos in range(self.header['DimSize'][1]) ]
        if axis == 'z':
            return [ float(self.header['Offset'][2]+(pos-hp)*self.header['ElementSpacing'][2]) for pos in range(self.header['DimSize'][2]) ]


    def coord2index(self,coord,halfpixel):
        assert(len(coord)==3)

        x_x = [x for x in self.get_axis_mms('x',halfpixel)]
        x_y = [x for x in self.get_axis_mms('y',halfpixel)]
        x_z = [x for x in self.get_axis_mms('z',halfpixel)]

        return [
            min(range(len(x_x)), key=lambda i: abs(x_x[i]-coord[0])),
            min(range(len(x_y)), key=lambda i: abs(x_y[i]-coord[1])),
            min(range(len(x_z)), key=lambda i: abs(x_z[i]-coord[2])),
        ]


    def save1dlist(self,outpostfix,crush):
        # assert crush.count(0) is 1

        # crush=crush[::-1]
        # outdata = self.imdata.reshape(self.imdata.shape[::-1]).copy()

        # ax = [i for (i,j) in zip(range(len(crush)),crush) if j==1]
        # outdata = np.add.reduce(outdata, axis=tuple(ax))
        outdata = self.get1dlist(crush)
        outname = self.file[:-4] + outpostfix
        with open(outname,'w') as thefile:
            pickle.dump(outdata.tolist(), thefile)
        return outname


    def getsum(self):
        return float(np.sum(self.imdata))


    def savesum(self,outpostfix):
        outname = self.file[:-4] + outpostfix
        with open(outname,'w') as thefile:
            pickle.dump(self.getsum(), thefile)
        return outname


    def tofake4d(self,binsInNewDim=250):
        #fake 4D that exploits an MHD-headertrick
        if self.header['NDims'] is not 3:
            print("Can only generate 4D image from 3D image.")
            return
        inraw = self.header['ElementDataFile'] #we will delete this value before we need to reuse
        self.header['DimSize'].append(binsInNewDim)
        self.header['NDims']+=1
        #self.header['TransformMatrix'] = [int(x) for x in '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'.split()]
        #self.header['CenterOfRotation'] = [int(x) for x in '0 0 0 0'.split()]
        self.header['Offset'].append(0)
        self.header['ElementSpacing'].append(1)
        self.header['ElementDataFile']='LIST'
        newhead = self.__getheaderasstring()
        for i in range(binsInNewDim):
            newhead.append(inraw)
        newheadfile = self.file[:-4]+'4d.mhd'
        with open(newheadfile,'w+') as newheadf:
            newheadf.writelines("%s\n" % l for l in newhead)
        print("New 4D mhd file:",newheadfile)
        print("This requires the original .raw file to be present.")
        return image(newheadfile)


    def applyfake4dmask(self,postfix,*maskfiles):
        inname = self.file
        outname = self.file[:-4]+str(postfix)+'.mhd'
        #print outname
        for maskfile in maskfiles:
            subprocess.call(['clitkImageArithm','-t 1','-i',inname,'-j',maskfile,'-o',outname])
            inname = outname
        print("New mhd file:",outname)
        return image(outname)


    def applymask_clitk(self,postfix,maskfile,p=0.):
        inname = self.file
        outname = self.file[:-4]+str(postfix)+'.mhd'
        subprocess.call(['clitkSetBackground','-i',inname,'-m',maskfile,'-o',outname,'-p',str(p)])
        #os.popen("clitkSetBackground -i "+inname+" -m "+maskfile+" -o "+outname)
        print("New mhd file:",outname)
        return image(outname)


    def smudge(self,mskval,frac=1.):
        #assume mskval must be ignored
        tmp = np.ma.masked_where(self.imdata == mskval, self.imdata)
        self.imdata[self.imdata != mskval] = tmp.mean()


    def applymask(self,*maskimages):
        for msk in maskimages:
            #reshape first to inverted axis order.
            mskcopy = msk.imdata.reshape(msk.imdata.shape[::-1])
            self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
            #I don't exactly know why, but then simple broadcasting,multiplication works
            self.imdata = mskcopy*self.imdata
            #doesnt seem necesarry, but lets convert back to correct dimensions anyway
            self.imdata = self.imdata.reshape(self.imdata.shape[::-1])


    def unionmask(self,*maskimages):
        for msk in maskimages:
            #reshape first to inverted axis order.
            mskcopy = msk.imdata.reshape(msk.imdata.shape[::-1])
            self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
            #I don't exactly know why, but then simple broadcasting,multiplication works
            self.imdata = np.logical_or(mskcopy,self.imdata).astype(int)
            #doesnt seem necesarry, but lets convert back to correct dimensions anyway
            self.imdata = self.imdata.reshape(self.imdata.shape[::-1])


    def savewithmask(self,outpostfix,*maskfiles):
        self.applymask(*maskfiles)
        return self.saveas(outpostfix)


    def save90pcmask(self):
        outpostfix = '.90pcmask'
        self.to90pcmask(outpostfix)
        return self.saveas(outpostfix)


    def to90pcmask(self,outpostfix='.90pcmask'):
        if self.header['NDims'] == 4:
            crush = [0,0,0,1]
            self.__crush(crush)
            self.__crush_header(outpostfix,crush)
        shape = self.imdata.shape # so we can go back later
        self.imdata = self.imdata.flatten() #so that we have 1 index
        sortedindices = np.argsort(self.imdata) #, axis=None) #in case we didnt flatten

        running_pc = 0.0
        target_pc = 0.9 * np.sum(self.imdata) #90% of total sum.
        index_90=len(self.imdata)-1 #we start at the bin with highest yield (end)
        while running_pc < target_pc:
            running_pc += self.imdata[sortedindices[index_90]];
            index_90-=1
        for i in range(len(sortedindices)):
            if i<=index_90: # below 90%, not interested
                self.imdata[sortedindices[i]] = 0
            elif i>index_90: # we want only what's above the 90% boundary
                self.imdata[sortedindices[i]] = 1
        self.imdata = np.reshape(self.imdata,shape) # puterback


    def toNpcmask(self,N,outpostfix='.90pcmask'):
        if self.header['NDims'] == 4:
            crush = [0,0,0,1]
            self.__crush(crush)
            self.__crush_header(outpostfix,crush)
        shape = self.imdata.shape # so we can go back later
        self.imdata = self.imdata.flatten() #so that we have 1 index
        sortedindices = np.argsort(self.imdata) #, axis=None) #in case we didnt flatten

        running_pc = 0.0
        target_pc = float(N)/100. * np.sum(self.imdata) #90% of total sum.
        index_90=len(self.imdata)-1 #we start at the bin with highest yield (end)
        while running_pc < target_pc:
            running_pc += self.imdata[sortedindices[index_90]];
            index_90-=1
        for i in range(len(sortedindices)):
            if i<=index_90: # below 90%, not interested
                self.imdata[sortedindices[i]] = 0
            elif i>index_90: # we want only what's above the 90% boundary
                self.imdata[sortedindices[i]] = 1
        self.imdata = np.reshape(self.imdata,shape) # puterback


    def todosemask(self,outpostfix='.dosemask'):
        np.where(self.imdata>0, 1, 0)


    def tomask_atthreshold(self,threshold):
        self.imdata[self.imdata<threshold] = 0
        self.imdata[self.imdata>=threshold] = 1


    def savemask_atthreshold(self,threshold,outpostfix='.masked'):
        self.tomask_atthreshold(threshold)
        return self.saveas(outpostfix)


    def tolowpass(self,threshold):
        self.imdata[self.imdata<threshold] = 0


    def tohighpass(self,threshold):
        self.imdata[self.imdata>threshold] = 0


    def savelowpass(self,threshold,outpostfix='.lowpass'):
        self.tolowpass(threshold)
        return self.saveas(outpostfix)


    def normalize(self):
        self.imdata = self.imdata/self.imdata.max()


    def savenormalize(self,outpostfix='.normalize'):
        self.normalize()
        return self.saveas(outpostfix)


    def savehighpass(self,threshold,outpostfix='.highpass'):
        self.tohighpass(threshold)
        return self.saveas(outpostfix)


    def toeff(self):
        runtime = self.nprim/self.pps
        if self.type == 'var' or self.type == 'relvar' or self.type == 'relunc' or  self.type == 'unc':
            with np.errstate(divide='ignore', invalid='ignore'):
                self.imdata = np.true_divide(1.,(runtime*self.imdata))
        #elif self.type == 'relunc':
        #	with np.errstate(divide='ignore', invalid='ignore'):
        #		self.imdata = np.true_divide(1.,(runtime*np.square(self.imdata)))
        else:
            print("Cant compute efficiency: this image is neither of variance or relative uncertainty!")
            return
        self.imdata[self.imdata == np.inf] = 0
        self.imdata[self.imdata == np.nan] = 0
        self.type = 'eff'


    def torelunc(self,yieldimage):
        assert self.type == 'var'
        assert yieldimage.type == 'yield'
        #print 'brent',self.nprim
        with np.errstate(divide='ignore', invalid='ignore'):
            self.imdata = np.true_divide(np.sqrt(self.imdata),yieldimage.imdata)
            #self.imdata = np.true_divide(np.sqrt(self.imdata),yieldimage.imdata*sqrt(self.nprim))
        self.imdata[self.imdata == np.inf] = 0
        self.imdata[self.imdata == np.nan] = 0
        self.type = 'relunc'


    def tounc(self):
        assert self.type == 'var'
        np.sqrt(self.imdata)
        self.type = 'unc'


    def tocount(self):
        if self.type == 'var':
            np.sqrt(self.imdata)*self.nprim
        if self.type == 'yield':
            np.sqrt(self.imdata)*self.nprim
        if self.type == 'unc':
            np.sqrt(self.imdata)*(self.nprim**2)
        if self.type == 'relunc':
            np.sqrt(self.imdata)*self.nprim


    def torelvar(self,yieldimage):
        assert self.type == 'var'
        assert yieldimage.type == 'yield'
        with np.errstate(divide='ignore', invalid='ignore'):
            self.imdata = np.true_divide(self.imdata,yieldimage.imdata)
        self.imdata[self.imdata == np.inf] = 0
        self.imdata[self.imdata == np.nan] = 0
        self.type = 'relvar'


    def toeffratio(self,otherimage):
        assert self.type == 'eff'
        assert otherimage.type == 'eff'
        with np.errstate(divide='ignore', invalid='ignore'):
            self.imdata = np.true_divide(self.imdata,otherimage.imdata)
        self.imdata[self.imdata == np.inf] = 0
        self.imdata[self.imdata == np.nan] = 0
        self.type = 'effratio'


    def gethighest(self):
        imdata = self.imdata.reshape(self.imdata.shape[::-1])
        return list(np.unravel_index(np.nanargmax(imdata),imdata.shape))[::-1]


    def getlowest(self):
        imdata = self.imdata.reshape(self.imdata.shape[::-1])
        return list(np.unravel_index(np.nanargmin(imdata),imdata.shape))[::-1]


    def getpixel(self,coord):
        dim = len(self.imdata.shape)
        imdata = self.imdata.reshape(self.imdata.shape[::-1])
        coord=coord[::-1]
        if dim is 4:
            return imdata[:,coord[0],coord[1],coord[2]]
        if dim is 3:
            return imdata[coord[0],coord[1],coord[2]]
        return #self.imdata.item(tuple(coord))


    def getmean(self):
        #return np.mean(self.imdata)
        return np.mean(self.imdata[self.imdata.nonzero()])


    def filter_bone(self):
        dim = len(self.imdata.shape)
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
        if dim is 4:
            self.imdata[:,10,:,:]=0
        if dim is 3:
            self.imdata[10,:,:] = 0
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])


    def filter_front(self):
        dim = len(self.imdata.shape)
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
        if dim is 4:
            self.imdata[:,0:70,:,:]=0
        if dim is 3:
            self.imdata[0:70,:,:] = 0
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])


    def insert_block_plusx(self,cutoff):
        dim = len(self.imdata.shape)
        assert dim is 3
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
        self.imdata[0:cutoff,:,:] = 0
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])


    def insert_block_minx(self,cutoff):
        dim = len(self.imdata.shape)
        assert dim is 3
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
        self.imdata[-cutoff:-1,:,:] = 0
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])


    def insert_block(self,ax,sign,cutoff,value=0):
        dim = len(self.imdata.shape)
        assert dim is 3
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])

        if 'x' in ax:
            if '-' in sign:
                self.imdata[-cutoff:-1,:,:] = value
                self.imdata[-1,:,:] = value		#dont ask dont tell
            if '+' in sign:
                self.imdata[0:cutoff,:,:] = value

        if 'y' in ax:
            if '-' in sign:
                self.imdata[:,-cutoff:-1,:] = value
                self.imdata[:,-1,:] = value
            if '+' in sign:
                self.imdata[:,0:cutoff,:] = value

        if 'z' in ax:
            if '-' in sign:
                self.imdata[:,:,-cutoff:-1] = value
                self.imdata[:,:,-1] = value
            if '+' in sign:
                self.imdata[:,:,0:cutoff] = value

        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])


    def divide_by_nprim(self):
        assert self.type == 'var'
        #divide once again thru nprim
        with np.errstate(divide='ignore', invalid='ignore'):
            self.imdata = np.true_divide(self.imdata,self.nprim)


    def divide(self,N):
        assert self.type == 'var'
        with np.errstate(divide='ignore', invalid='ignore'):
            self.imdata = np.true_divide(self.imdata,N)


    def unctovar(self):
        assert self.type == 'unc'
        self.imdata = np.square(self.imdata)
        self.type = 'var'


    def clip_range(self,mini,maxi):
        assert self.datatype == '<i2'
        np.clip(self.imdata, mini, maxi,out=self.imdata)


    def saveslice(self,ax,ind):
        backup = np.copy(self.imdata)
        backuph = np.copy(self.header)
        self.__crush([0,0,0,1]) #to 3d
        outpostfix = '.slice'+ax+str(ind)

        #why-o-why but it works
        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])

        if ax is 'x':
            #self.imdata = self.imdata[ind,:,:]
            self.imdata = self.imdata[:,:,ind]
            self.__crush_header(outpostfix,[1,0,0,1])
        if ax is 'y':
            self.imdata = self.imdata[:,ind,:]
            self.__crush_header(outpostfix,[0,1,0,1])
        if ax is 'z':
            #self.imdata = self.imdata[:,:,ind]
            self.imdata = self.imdata[ind,:,:]
            self.__crush_header(outpostfix,[0,0,1,1])

        self.imdata = self.imdata.reshape(self.imdata.shape[::-1])
        print(self.imdata.shape)

        self.saveas(outpostfix)
        r = self.imdata

        self.imdata = backup
        self.header = backuph
        self.crushed = False
        return r


    def getcenter(self,spotsize=5):
        pica = self.imdata.squeeze()
        assert len(pica.shape) == 2
        mx = pica.shape[1]/2
        my = pica.shape[-1]/2
        hs = spotsize/2
        return pica[ mx-hs : mx+hs , my-hs : my+hs ]


    def getslice(self,ax,ind):
        outslice = self.imdata.reshape(self.imdata.shape[::-1])
        if ax is 'x':
            outslice = outslice[:,:,ind]
        if ax is 'y':
            outslice = outslice[:,ind,:]
        if ax is 'z':
            outslice = outslice[ind,:,:]

        #outslice = outslice.reshape(outslice.shape[::-1])
        print(outslice.shape)
        return outslice


    def togammamask(self):
        #convert to image that is of same size, and filters out <1 >8 MeV gammas
        #this assumes the full range corresponds to 1-8MeV
        assert self.header['NDims'] == 4
        dim = self.header['DimSize']
        shape = self.imdata.shape
        self.imdata = np.zeros(shape[::-1],dtype='<u1')
        print('Converted to empty image of shape',shape)
        self.header['ElementType'] = 'MET_UCHAR'
        startbin = int(1./10.*dim[-1]) #dim[-1] == 250 usually
        endbin = int(8./10.*dim[-1])+1 #let's include 8MeV to be consistent with old mask.
        #self.imdata[:,:,:,startbin:endbin] = 1
        self.imdata[startbin:endbin,:,:,:] = 1
        self.imdata = self.imdata.reshape(shape)


    def towater(self):
        self.imdata.fill(0)


    def savegammamask(self):
        self.togammamask()
        self.saveas('1-8msk')
