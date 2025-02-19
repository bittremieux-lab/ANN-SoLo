import codecs
import os
import setuptools

import numpy as np

try:
    import Cython.Distutils
except ImportError:
    use_cython = False
else:
    use_cython = True


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # Intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string')


DISTNAME = 'ann_solo'
# https://packaging.python.org/guides/single-sourcing-package-version/
VERSION = get_version('ann_solo/__init__.py')
DESCRIPTION = 'Spectral library search engine optimized for fast open ' \
              'modification searching'
with open('README.md') as f_in:
    LONG_DESCRIPTION = f_in.read()
AUTHOR = 'Wout Bittremieux'
AUTHOR_EMAIL = 'wout.bittremieux@uantwerpen.be'
URL = 'https://github.com/bittremieux/ANN-SoLo'
LICENSE = 'Apache 2.0'


compile_args = ['-O3', '-march=native', '-ffast-math', '-fno-associative-math',
                '-std=c++14']
cython_directives = {'boundscheck': False, 'wraparound': False,
                     'initializedcheck': False, 'language_level': 3}
ext_spectrum_match.cython_directives = cython_directives
ext_parsers = setuptools.Extension(
    'ann_solo.parsers', ['ann_solo/parsers.pyx'],
    language='c++', extra_compile_args=compile_args,
    extra_link_args=compile_args, include_dirs=[np.get_include()])
ext_parsers.cython_directives = cython_directives

cmdclass = {}
if use_cython:
    cmdclass.update({'build_ext': Cython.Distutils.build_ext})

setuptools.setup(
    name=DISTNAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url=URL,
    license=LICENSE,
    platforms=['any'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
    packages=['ann_solo'],
    entry_points={
        'console_scripts': ['ann_solo = ann_solo.ann_solo:main',
                            'ann_solo_plot = ann_solo.plot_ssm:main']},
    cmdclass=cmdclass,
    install_requires=[
        'ConfigArgParse',
        'Cython',
        'faiss',
        'joblib',
        'matplotlib',
        'mmh3',
        'mokapot>=v0.8.3',
        'numba>=0.41',
        'numexpr',
        'numpy',
        'pandas',
        'pyteomics',
        'scipy',
        'spectrum_utils>=0.4.2',
        'tqdm'],
    setup_requires=[
        'Cython',
        'numpy'],
    ext_modules=[ext_spectrum_match, ext_parsers],
)
