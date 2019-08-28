#! /usr/bin/env python
from setuptools import setup

VERSION = '1.0.0'

with open("README.md", "rb") as f:
    long_descr = f.read()

def main():
    setup(name='medimage',
          version=VERSION,
          description="Represent medical images as numpy array. Supported: .mhd (R/W),.xdr (R/W), dicom (R). Pure Python.",
          long_description=open('README.md').read(),
          classifiers=[
              'Development Status :: 5 - Production/Stable',
              'Environment :: Console',
              'Environment :: MacOS X',
              'Environment :: Win32 (MS Windows)',
              'Environment :: X11 Applications',
              'Programming Language :: Python :: 3',
              'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
              'Topic :: Utilities',
              'Operating System :: OS Independent'
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
