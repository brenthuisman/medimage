## medimage

This library supports r/w MetaImage (MHD,ITK), r/w AVSField (.xdr) and read Dicom images. XDR reading includes NKI compressed images (useful to work with your Elekta images). The `image` class is a thin wrapper around typed numpy array objects (the `.imdata` member) such that you can easily work with images in these data formats. Slicing, projections, mathematical operations, masking, stuff like that is very easy with numpy, so you can easily extend things to what you need.

Included are some basic mathematical operations, some masking functions and crop and resampling functions. Of particular interest perhaps are the DVH analysis function, and the distance to agreement calculation. This calculation is quite slow though. For [NKI decompression](https://gitlab.com/plastimatch/plastimatch/tree/master/libs/nkidecompress) I supply a 64bit Linux and Windows lib, if you need support for other platforms you can compile the function in `medimage/nki_decomp` yourself. This component is governed by its own license.

Dicom write is not supported right now. If it would, it would *require* `SimpleITK`, primarily because `pydicom` does not support dicom image write... SimpleITK write also only seems to produce usable dicoms files when updating an existing image, not when creating a new one from scratch.


## Motivation

This project started out at a time when I was analyzing lots of [Gate](https://github.com/opengate/gate) image outputs. ITK's Python bindings (SimpleITK) was not pippable or easily usable yet, and I found working with image data as numpy arrays far preferable and faster than using ITK as a library in custom C++ programs which I'd need to compile and recompile as an analysis developed. Matplotlib after all is Python-only.

I wanted to have a thin and pure Python wrapper around `numpy` that would allows me to read in and write out image data. Fortunately, the (uncompressed) MetaImage disk format was so straightforward even I could understand it, and it was even suprisingly performant. This `image` class grew to suit my needs as part of my `phd_tools` and later `postdoc_tools` repos, and in a new job I ported it to Python 3 and added filesupport for AVSFields and Dicom images. The idea is that you can take the `medimage` directory, drop it into any project, and be able to work with medical images as numpy arrays. It is now a core component to my analyses, and perhaps it can be useful to you too. A lot of machine learning tooling are heavy users of `numpy`, and therefore getting your images in is straightforward with this package.

## Install

You can now use pip!

    $ pip3 install medimage (--user)

Or clone/download this repo and install manually with:

    $ python3 setup.py install (--user)

Currently, the `pymedphys` component is NOT installed automatically, which is required when you are going to use the `compute_gamma` method. That is because it is a rather large package, and in developmental flux.

## Usage

After installation, you should be able to open and save an image like so:

	from medimage import image
	myfirstimage = image("somefile.xdr")
	myfirstimage.saveas("somefile.mhd")

`image`s are instatiated with a string representing a file location, where the file extension indicates filetype. If not known extension is found, it assumes you're providing a dicom image or a dicom directory (of images).

Alternatively, you can make a new zeroed out image of 30 by 40 by 50 voxels, spaced out 2mm in each dimension, centered at zero, like so:

	from medimage import image
	myblankimage = image(DimSize=[30,40,50],ElementSpacing=[2,2,2],Offset=[0,0,0])
	myblackimage.saveas("empty.mhd")

## Coordinates

I've taken great care to make sure this library can work with images of any dimensionality, and that your image as represented by the `.imdata` numpy array member, and any class methods, have straightforward indexing (e.g. [x,y,z,], not [z,y,x]). PRs that fix any bugs in this regard are very welcome.

### Take a slice or line profile

You may want to look at or work with a slice. Let's say you have a 3D image at `fname`:

	image = image.image(fname)
	x,y,z=image.get_slices_at_index() #defaults to central voxel
	import scipy.misc
	scipy.misc.imsave("d:/slicex.png",x)

Need to look at a profile?

	image.get_profiles_at_index([10,10,10]) #get the lines through voxel [10,10,10]

Don't care about indeces, but you know in physical dimensions where you want to slice? Say, through the point at 23.4mm,10mm,2.3mm?

	image.get_pixel_index([23.4,10,2.3])

## Cropping

Apart from regular old cropping, the `.crop_as` method let's you 'crop' an image to the same size and pixelspacing as another image, provided that the images have overlap. For instance, you may have a CT and a dose image, where the dose image has larger voxel and covers only a subregion of the CT. You can get your CT values for each dose voxel like so:

	ct = image.image('ct.dcm')
	dose = image.image('dose.dcm')
	ct.crop_as(dose)
	ct.saveas('ct_dosegrid.xdr')

### DVH parameters within a subregion for which you have a mask

Say you have a dose calculation and you want to have some DVH metrics (say, Dmax,D2,D50,D98,Dmean). Suppose you want those DVH metrics in the PTV region, and you have a PTV as mask image. How could `medimage` do this for you?

	from medimage import image
	import argparse
	from os import path

	parser = argparse.ArgumentParser(description='Supply an image and a mask or percentage for isodose contour in which to compute the DVH.')
	parser.add_argument('inputimage')
	parser.add_argument('--maskimage',default=None)
	parser.add_argument('--maskregion',default=None,type=float)
	opt = parser.parse_args()

	im = image.image(path.abspath(opt.inputimage))
	maskim = None

	if opt.maskregion == None and path.isfile(path.abspath(opt.maskimage)):
		print('Using',opt.maskimage,'as region for DVH analysis.')
		maskim = image.image(path.abspath(opt.maskimage))
	elif opt.maskregion != None:
		assert 0 < opt.maskregion < 100
		print('Using isodose contour at',opt.maskregion,'percent of maximum dose as region for DVH analysis.')
		maskim = im.copy()
		maskim.tomask_atthreshold((opt.maskregion/100.)*maskim.max())
	else:
		print('No mask or maskregion specified; using whole volume for DVH analysis.')

	if maskim != None:
		im.applymask(maskim)

	# note: array is sorted in reverse for DVHs, i.e. compute 100-n%
	D2,D50,D98 = im.percentiles([98,50,2])

	print("Dmax,D2,D50,D98,Dmean")
	print(im.imdata.max(),D2,D50,D98,im.mean())

## Dependencies

 * numpy
 * pydicom

### Changelog

 * 2020-02-13: v1.0.7: Bugfixes
 * 2019-10-08: v1.0.6: Bugfix, dicom write still incomplete.
 * 2019-10-08: v1.0.5: Dicom write
 * 2019-09-24: v1.0.4: New and much faster gamma computation (order of 5 minutes)
 * 2019-08-28: v1.0.3: Fixed a few sloppy bugs. Added CT rescaling when openingen Dicom image.
 * 2019-08-28: v1.0.0: Separated `image` class into its own `medimage` module.
