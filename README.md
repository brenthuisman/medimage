## medimage

This library supports r/w MetaImage (MHD,ITK) and r/w AVSField (.xdr) images, including NKI compressed images (read-only, useful to work with your Elekta images). The `image` class is a thin wrapper around typed numpy array objects (the `imdata` member) such that you can easily work with images in these data formats. Slicing, projections, mathematical operations, masking, stuff like that is very easy with numpy, so you can easily extend things to what you need.

Included are some basic mathematical operations, some masking functions and crop and resampling functions. Of particular interest perhaps are the DVH analysis function, and the distance to agreement calculation (entirely based on the `gamma` component in [pymedphys](https://github.com/pymedphys/pymedphys). This calculation is quite slow though. For [NKI decompression](https://gitlab.com/plastimatch/plastimatch/tree/master/libs/nkidecompress) I supply a 64bit Linux and Windows lib, if you need support for other platforms you can compile the function in `image/nki_decomp` yourself. This component is governed by its own license.

## Motivation

This project started out at a time when I was analyzing lots of [Gate](https://github.com/opengate/gate) image outputs. ITK's Python bindings (SimpleITK) was not pippable or easily usable yet, and I found working with image data as numpy arrays far preferable and faster than using ITK as a library in custom C++ programs which I'd need to compile and recompile as an analysis developed. Matplotlib after all is Python-only.

I wanted to have a thin and pure Python wrapper around `numpy` that would allows me to read in and write out image data. Fortunately, the (uncompressed) MetaImage disk format was so straightforward even I could understand it, and it was even suprisingly performant. This `image` class grew to suit my needs as part of my `phd_tools` and later `postdoc_tools` repos, and in a new job I ported it to Python 3 and added filesupport for AVSFields and Dicom images. The idea is that you can take the `image` directory, drop it into any project, and be able to work with medical images as numpy arrays. It is now a core component to my analyses, and perhaps it can be useful to you too.

## Install

You can now use pip!

    $ pip(3) install medimage (--user)

Or clone/download this repo and install manually with:

    $ python(3) setup.py install (--user)

Currently, the `pymedphys` component is NOT installed automatically, which is required when you are going to use the `compute_gamma` method. That is because it is a rather large package, and in developmental flux.

## Usage

After installation, you should be able to instantiate `image` classes like so:

	from medimage import image
	myfirstimage = image("somefile.xdr")

`image`s are instatiated with a string representing a file location, where the file extension indicates filetype. If not known extension is found, it assumes you're providing a dicom image or a dicom directory (of images).

Alternatively, you can make a new zeroed out image of 30 by 40 by 50 voxels, spaced out 2mm in each dimension, centered at zero, like so:

	from medimage import image
	myblankimage = image.(DimSize=[30,40,50],ElementSpacing=[2,2,2],Offset=[0,0,0])

An optional `par2deep.ini` file may be placed in the target directory or as `~/.par2deep` defining all the commandline options. For the excludes, separate by comma.

## Some DVH parameters

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
 * optional: pymedphys

### Changelog

 * 2019-08-28: v1.0.0: Separated `image` class into its own `medimage` module.
