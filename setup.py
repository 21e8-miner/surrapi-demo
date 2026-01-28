from setuptools import setup, find_packages

setup(
    name="surrapi",
    version="0.1.0",
    description="Python SDK for SurrAPI - Neural CFD Surrogate",
    author="SurrAPI Team",
    author_email="team@surrapi.io",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
        "numpy>=1.22.0",
        "pydantic>=2.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
