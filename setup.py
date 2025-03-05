from setuptools import setup, find_packages

setup(
    name="python_transdecoder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "biopython>=1.79",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "click>=8.0.0",  # For command line interface
    ],
    entry_points={
        "console_scripts": [
            "transdecoder-longorfs=transdecoder.longorfs:main",
            "transdecoder-predict=transdecoder.predict:main",
        ],
    },
    author="TransDecoder Project",
    author_email="example@example.com",
    description="Python port of TransDecoder for identifying coding regions in transcripts",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TransDecoder/TransDecoder",
    python_requires=">=3.8",
)