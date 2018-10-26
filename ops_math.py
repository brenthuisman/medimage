import numpy as np

class math_class:
    def getsum(self):
        return np.sum(self.imdata)


    def getmean(self):
        #return np.mean(self.imdata)
        return np.mean(self.imdata[self.imdata.nonzero()])


    def smudge(self,mskval,frac=1.):
        #assume mskval must be ignored
        tmp = np.ma.masked_where(self.imdata == mskval, self.imdata)
        self.imdata[self.imdata != mskval] = tmp.mean()


    def tolowpass(self,threshold):
        self.imdata[self.imdata<threshold] = 0


    def tohighpass(self,threshold):
        self.imdata[self.imdata>threshold] = 0


    def normalize(self):
        self.imdata = self.imdata/self.imdata.max()


    def divide(self,N):
        assert self.type == 'var'
        with np.errstate(divide='ignore', invalid='ignore'):
            self.imdata = np.true_divide(self.imdata,N)


    def clip_range(self,mini,maxi):
        assert self.datatype == '<i2'
        np.clip(self.imdata, mini, maxi,out=self.imdata)


    def gethighest(self):
        '''get index of highest value in image'''
        return list(np.unravel_index(np.nanargmax(self.imdata),self.imdata.shape))


    def getlowest(self):
        '''get index of lowest value in image'''
        return list(np.unravel_index(np.nanargmin(self.imdata),self.imdata.shape))
