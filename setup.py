import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['deep_daze']
from version import __version__

setup(
  name = 'deep-daze',
  packages = find_packages(),
  include_package_data = True,
  package_data = {'bpe': ['data/bpe_simple_vocab_16e6.txt']},
  entry_points={
    'console_scripts': [
      'imagine = deep_daze.cli:main',
    ],
  },
  version = __version__,
  license='MIT',
  description = 'Deep Daze',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/deep-daze',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'fire',
    'transformers',
    'implicit neural representations',
    'text to image'
  ],
  install_requires=[
    'torch>=1.7.1',
    'einops>=0.3',
    'ftfy',
    'numpy',
    'siren-pytorch>=0.0.6',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
