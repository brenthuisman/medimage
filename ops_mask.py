import numpy as np
import subprocess, sys

class mask_class:
    def towater(self):
        self.imdata.fill(0)

    def tofake4d(self,binsInNewDim=250):
        # TODO: remove?
        #fake 4D that exploits an MHD-headertrick
        if self.header['NDims'] is not 3:
            print("Can only generate 4D image from 3D image.",file=sys.stderr)
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
        print("New 4D mhd file:",newheadfile,file=sys.stderr)
        print("This requires the original .raw file to be present.",file=sys.stderr)
        return image(newheadfile)


    def applymask(self,*maskimages):
        ''' Applies the mask(s) you specify to imdata.'''
        for msk in maskimages:
            assert type(msk) == type(self)
            self.imdata = np.ma.masked_array(self.imdata,mask=msk.imdata,fill_value=np.nan)


    def unionmask(self,*maskimages):
        for msk in maskimages:
            assert type(msk) == type(self)
            self.imdata = np.logical_or(msk,self.imdata).astype(int)


    def tomask_atvolume(self,N=90):
        ''' Make this image a mask based on the dose higher than the threshold of total dose (as percentage) you specify.
        i.e. generates isodose mask at given volume in DVH. '''

        shape = self.imdata.shape # so we can go back later
        self.imdata = self.imdata.flatten() #so that we have 1 index
        sortedindices = np.argsort(self.imdata) #, axis=None) #in case we didnt flatten

        # running total of total sum is cumulatief distribution, i.e. DVH
        running_pc = 0.0
        target_pc = float(N)/100. * np.sum(self.imdata) #N% of total sum.
        index_N=len(self.imdata)-1 #we start at the bin with highest yield (end)
        while running_pc < target_pc:
            running_pc += self.imdata[sortedindices[index_N]]
            index_N-=1
        for i in range(len(sortedindices)):
            if i<=index_N: # below 90%, not interested
                self.imdata[sortedindices[i]] = 1
            elif i>index_N: # we want only what's above the 90% boundary
                self.imdata[sortedindices[i]] = 0
        self.imdata = np.reshape(self.imdata,shape) # puterback


    def tomask_atthreshold(self,threshold=None,invert=True):
        ''' Makes makes at requested threshold '''

        if threshold==None:
            threshold=float(self.imdata.max())*0.5

        if invert:
            self.imdata[self.imdata<threshold] = 1
            self.imdata[self.imdata>=threshold] = 0
        else:
            self.imdata[self.imdata<threshold] = 0
            self.imdata[self.imdata>=threshold] = 1
