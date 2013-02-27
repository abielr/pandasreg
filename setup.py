from distutils.core import setup, Extension
from numpy.distutils import misc_util
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# ext_modules = [
# 	Extension("pandasreg.rperiodlib", ["pandasreg/src/rperiodlib.pyx"])
# 	Extension("pandasreg.rfreq", ["pandasreg/src/rfreq.pyx"])
# ]

# ext_modules = cythonize("pandasreg/src/*.pyx")
ext_modules = cythonize([
	Extension("pandasreg.rfreq", ["pandasreg/src/rfreq.pyx"])
])

setup(
    name = 'pandasreg',
	version='0.1.0',
	author='Abiel Reinhart',
	author_email='abielr@gmail.com',
	packages=['pandasreg','pandasreg.test'],
	url='http://www.github.com/abielr/pandasreg',
	license='LICENSE.txt',
	description='Pandas extensions for regularly-spaced time series',
	long_description=open('README.txt').read(),
	install_requires=[
		"pandas >= 0.10.1"
	],

    cmdclass = {'build_ext': build_ext},
    include_dirs = misc_util.get_numpy_include_dirs()+['pandasreg/src'],
    ext_modules = ext_modules
)