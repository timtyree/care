#!/usr/bin/env python3
#!/usr/bin/python
from setuptools import setup
from Cython.Build import cythonize
from skimage._build import cython
import os
base_path = os.path.abspath(os.path.dirname(__file__))

############################################
# Example Usage - compile from command line with
# $ python setup.py build_ext --inplace
############################################

def configuration(parent_package='', top_path=None):
		from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

		config = Configuration('measure', parent_package, top_path)

		cython(['_find_contours_cy.pyx'
				,'_find_contours_pbc_cy.pyx' 
				,'_find_tips_pbc_cy.pyx'
				,'_find_tips_kernel_cy.pyx'
				], 
				working_path=base_path)

		# config.add_extension('_ccomp', sources=['_ccomp.c'],
		#                      include_dirs=[get_numpy_include_dirs()])
		config.add_extension('_find_contours_cy', sources=['_find_contours_cy.c'],
												 include_dirs=[get_numpy_include_dirs()])
		config.add_extension('_find_contours_pbc_cy', sources=['_find_contours_pbc_cy.c'],
												 include_dirs=[get_numpy_include_dirs()])
		config.add_extension('_find_tips_pbc_cy', sources=['_find_tips_pbc_cy.c'],
												 include_dirs=[get_numpy_include_dirs()])
		config.add_extension('_find_tips_kernel_cy', sources=['_find_tips_kernel_cy.c'],
												 include_dirs=[get_numpy_include_dirs()])
		return config

if __name__ == '__main__':
		from numpy.distutils.core import setup
		setup(maintainer='care Developers',
					maintainer_email=None,#'care@python.org',
					description='Lewiner marching squares (with optional periodic boundary conditions)',
					url=None,#'https://github.com/scikit-image/scikit-image',
					license='Modified MIT',
					**(configuration(top_path='').todict())
					)
