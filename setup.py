import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ebpca",
    version="0.0.1.dev1",
    author="Xinyi Zhong, Chang Su",
    author_email="zhongxy14@gmail.com, c.su@yale.edu",
    maintainer="Xinyi Zhong",
    maintainer_email="zhongxy14@gmail.com",
    description="implementation of EB-PCA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TraceyZhong/generalAMP",
    license="LICENSE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'matplotlib', 'scipy'],
    project_urls = {
        "Source": "https://github.com/TraceyZhong/generalAMP",
        "Paper": "NONWHERE",
    },
)