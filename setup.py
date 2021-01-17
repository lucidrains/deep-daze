from setuptools import setup, find_packages

setup(
  name = 'deep-daze',
  packages = find_packages(),
  version = '0.0.1',
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
    'torch>=1.7.1',
    'einops>=0.3'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
