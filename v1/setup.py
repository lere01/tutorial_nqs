from setuptools import setup, find_packages

setup(
    name='nqs_tutorial',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'jax',
        'flax',
        'numpy',
        'matplotlib',
        'notebook'
    ],
)
