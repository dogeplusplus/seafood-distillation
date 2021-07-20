from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name='seafood-distillation',
    version='',
    packages=find_packages(),
    install_requires=requirements,
    url='',
    license='',
    author='albert',
    author_email='',
    description=''
)
