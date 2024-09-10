import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='treegp',
    version='0.0.1',
    author='Luigi Rovito',
    author_email='luigirovito2@gmail.com',
    url='https://github.com/lurovi/TreeGP',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['wheel'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.11',
)
