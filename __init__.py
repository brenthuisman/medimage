import numpy as np,pickle,subprocess,os
from collections import OrderedDict
from decimal import Decimal
from functools import reduce

#convention: save...() functions return string with filename.mhd of new file.

class image:
	def __init__(self, infile, **kwargs):
		assert infile.endswith('.mhd')
		for key in ('pps', 'nprim', 'type'):
			if key in kwargs:
				setattr(self, key, kwargs[key])

		self.crushed = False
		self.infile = infile
		self.path = os.path.split(infile)[0]
		
		headerfile = open(infile,'r')
		self.header = OrderedDict()
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
				print("No valid input file (compressed).")
				return
			if 'DimSize' in newline[0]:
				self.header['DimSize'] = [int(x) for x in newline[1].split()]
			#if 'ElementDataFile' in newline[0]:
			#	inraw = newline[1]
			if 'ElementType' in newline[0]:
				self.header['ElementType'] = newline[1]
				if 'MET_FLOAT' in newline[1]:
					self.datatype = '<f4'
				if 'MET_DOUBLE' in newline[1]:
					self.datatype = '<f8'
				if 'MET_UCHAR' in newline[1]:
					self.datatype = '<u1'	
				if 'MET_SHORT' in newline[1]:
					self.datatype = '<i2'
				if 'MET_INT' in newline[1]:
					self.datatype = '<i4'
				if 'MET_LONG' in newline[1]:
					self.datatype = '<i8'
			if 'NDims' in newline[0]:
				self.header['NDims'] = int(newline[1])
			if 'TransformMatrix' in newline[0]:
				self.header['TransformMatrix'] = [Decimal(x) for x in newline[1].split()]
			if 'CenterOfRotation' in newline[0]:
				self.header['CenterOfRotation'] = [Decimal(x) for x in newline[1].split()]
			if 'Offset' in newline[0]:
				self.header['Offset'] = [Decimal(x) for x in newline[1].split()]
			if 'ElementSpacing' in newline[0]:
				self.header['ElementSpacing'] = [Decimal(x) for x in newline[1].split()]
			
		#print self.header['ElementDataFile'] #might have to replace with line below
		#self.inraw = infile[:-4]+".raw" #overwrite because of missing path
		if 'LIST' in self.header['ElementDataFile']:
			print("We have a fake 4D file, assuming 3D...")
			self.header['ElementDataFile'] = list(self.header.items())[-1][0]
			self.header['NDims'] -= 1
			self.header['DimSize'].pop()
		self.__loadimage()


	def __loadimage(self):
		#dt = '<f4' #np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('t','<f4')])
		dt = self.datatype
		dim = self.header['DimSize']
		indata = np.fromfile(os.path.join(self.path,self.header['ElementDataFile']), dtype=dt)
		assert len(indata) == reduce(lambda x, y: x*y, dim)
		self.nrvox=reduce(lambda x, y: x*y, dim[:-1])
		#print dim,len(indata)
		self.imdata = np.reshape(indata,tuple(dim))
		
		#correct for number of primaries.
		# try:
		# 	if 'var' in self.type:
		# 		self.imdata = self.imdata*(self.nprim**2)
		# 	elif 'yield' in self.type:
		# 		self.imdata = self.imdata*self.nprim
		# except AttributeError:
		# 	#nprim or type not set, no biggy
		# 	pass

		#opt: set inf and nan to zero
		#self.imdata[self.imdata == np.inf] = 0
		#self.imdata[self.imdata == np.nan] = 0

		#https://en.wikipedia.org/wiki/Row-major_order, dus achterstevoren dimensies geven.
		print(self.infile,"loaded. Shape:",self.imdata.shape)
		

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
		self.header['ElementDataFile'] = self.infile[:-4] + outpostfix + '.raw'

		#print self.header['ElementDataFile']

		try:
			self.header.pop('TransformMatrix') #FIXME: if ever needed.
			self.header.pop('CenterOfRotation')
		except KeyError:
			pass

		if self.header['NDims'] is 1:
			print("MHD doesnt support 1 dimensional images!")
			return


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
		newheadstr = []
		for k,v in list(newhead.items()):
			if v is None:
				newheadstr.append(str(k))
			else:
				newheadstr.append(str(k)+' = '+str(v))
		return newheadstr
	

	def saveas(self,outpostfix):
		outraw = self.infile[:-4] + outpostfix + '.raw'
		self.header['ElementDataFile'] = outraw
		if '/' in outraw:
			self.header['ElementDataFile'] = outraw.split('/')[-1]
		newheadfile = self.infile[:-4] + outpostfix + '.mhd'
		
		# VV doesnt support long, so we convert to int
		if self.imdata.dtype == np.int64:
			self.imdata = self.imdata.astype(np.int32, copy=False)
			self.datatype = '<i4'
			self.header['ElementType'] = 'MET_INT'
		
		#tofile is Row-major ('C' order), so that's why it happens to go correctly w.r.t. the HZYX order.
		self.imdata.tofile(outraw)
		print("New raw file:",outraw)
		if self.header['NDims'] is not 1:
			self.header['TotalSum'] = self.imdata.sum()
			with open(newheadfile,'w+') as newheadf:
				newheadf.writelines("%s\n" % l for l in self.__getheaderasstring())
			print("New mhd file:",newheadfile)
		return newheadfile


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
		pgbins = self.imdata.shape[-1]

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


	def save1dlist(self,outpostfix,crush):
		# assert crush.count(0) is 1

		# crush=crush[::-1]
		# outdata = self.imdata.reshape(self.imdata.shape[::-1]).copy()
		
		# ax = [i for (i,j) in zip(range(len(crush)),crush) if j==1]
		# outdata = np.add.reduce(outdata, axis=tuple(ax))
		outdata = self.get1dlist(crush)
		outname = self.infile[:-4] + outpostfix
		with open(outname,'w') as thefile:
			pickle.dump(outdata.tolist(), thefile)
		return outname
		

	def getsum(self):
		return float(np.sum(self.imdata))


	def savesum(self,outpostfix):
		outname = self.infile[:-4] + outpostfix
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
		newheadfile = self.infile[:-4]+'4d.mhd'
		with open(newheadfile,'w+') as newheadf:
			newheadf.writelines("%s\n" % l for l in newhead)
		print("New 4D mhd file:",newheadfile)
		print("This requires the original .raw file to be present.")
		return image(newheadfile)
		

	def applyfake4dmask(self,postfix,*maskfiles):
		inname = self.infile
		outname = self.infile[:-4]+str(postfix)+'.mhd'
		#print outname
		for maskfile in maskfiles:
			subprocess.call(['clitkImageArithm','-t 1','-i',inname,'-j',maskfile,'-o',outname])
			inname = outname
		print("New mhd file:",outname)
		return image(outname)


	def applymask_clitk(self,postfix,maskfile,p=0.):
		inname = self.infile
		outname = self.infile[:-4]+str(postfix)+'.mhd'
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
