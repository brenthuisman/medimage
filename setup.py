#! /usr/bin/env python
from setuptools import setup

VERSION = '1.0.3'

def main():
    setup(name='medimage',
          version=VERSION,
          description="Represent medical images as numpy array. Supported: .mhd (R/W),.xdr (R/W), dicom (R). Pure Python.",
          long_description=open('README.md').read(),
          long_description_content_type="text/markdown",
          classifiers=[
              'Development Status :: 5 - Production/Stable',
              'Programming Language :: Python :: 3',
              'Operating System :: OS Independent',
              'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
              "Topic :: Scientific/Engineering :: Medical Science Apps.",
              "Topic :: Scientific/Engineering :: Physics",
              "Intended Audience :: Science/Research",
              "Intended Audience :: Healthcare Industry",
          ],
          keywords='medical image xdr mhd dicom numpy',
          author='Brent Huisman',
          author_email='mail@brenthuisman.net',
          url='https://github.com/brenthuisman/medimage',
          license='LGPL',
          include_package_data=True,
          zip_safe=False,
          install_requires=['numpy','pydicom'],
          packages=['medimage'],
          )

if __name__ == '__main__':
    main()
