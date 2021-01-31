import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['deep_daze']
from version import __version__

setup(
  name = 'deep-daze',
  packages = find_packages(),
  include_package_data = True,
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
    'transformers',
    'implicit neural representations',
    'text to image'
  ],
  install_requires=[
    'einops>=0.3',
    'fire',
    'ftfy',
    'siren-pytorch>=0.0.8',
    'torch>=1.7.1',
    'torch_optimizer',
    'torchvision>=0.8.2',
    'tqdm',
    'regex'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
