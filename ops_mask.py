import numpy as np
import subprocess

class mask_class:
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
            running_pc += self.imdata[sortedindices[index_90]]
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

