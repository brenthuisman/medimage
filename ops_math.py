import numpy as np

class math_class:
    def size(self):
        return self.imdata.size

    def sum(self):
        return np.ma.sum(self.imdata)

    def max(self):
        return np.ma.max(self.imdata)

    def min(self):
        return np.ma.min(self.imdata)

    def mean(self):
        return np.ma.mean(self.imdata)

    def mean_excl_zero(self):
        return np.ma.mean(self.imdata[self.imdata.nonzero()])

    def median(self):
        return np.ma.median(self.imdata)


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


    def calc_gamma(self,other,dta,dd):
        assert type(other)==type(self)

        dd = dd * self.imdata.max()

        from npgamma import calc_gamma

        retval = self.copy()
        retval.imdata = calc_gamma(tuple(self.get_axes_labels()), self.imdata,tuple(other.get_axes_labels()), other.imdata, dta, dd, 0, dta / 3, dta*2, np.inf, 16)
        return retval

