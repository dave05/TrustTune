from setuptools import setup, find_packages

setup(
    name="trusttune",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "fastapi>=0.68.0",
    ],
    extras_require={
        "monitoring": ["psutil>=5.8.0"],
    },
) 