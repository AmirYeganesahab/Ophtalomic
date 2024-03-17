from distutils import extension
from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# to run this code in terminal : python3 setup.py build_ext --inplace

main_path = '.'
ext_modules = [
  Extension("gallery",[f'{main_path}/gallery.pyx']),
    Extension("classes",[f'{main_path}/classes.pyx']),
    Extension("sensors",[f'{main_path}/sensors.pyx']),
    Extension("main",[f'{main_path}/main.pyx']),
    Extension("tensorrt_code",[f'{main_path}/tensorrt_code.pyx'])

]
for e in ext_modules:
    e.cython_directives = {'language_level': "3"}
# extensions = cythonize(ext_modules,language_level = "3")

# setup(ext_modules= extensions)

setup(
  name = 'XcameraApp',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)