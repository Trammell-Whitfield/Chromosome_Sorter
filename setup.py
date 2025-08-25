#!/usr/bin/env python3
"""
Setup script for Chromosome Karyotype Sorter
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="chromosome-sorter",
    version="1.0.0",
    description="AI-powered chromosome karyotype sorting system",
    long_description="""
    A machine learning system that automatically sorts chromosomes in karyotype images.
    Uses both classification and regression approaches to identify and arrange chromosomes
    according to their morphological features.
    
    Problem: Scientists must manually arrange chromosomes into karyotypes, which is tedious.
    Solution: ML-based classification with regression fallback for automated sorting.
    """,
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/chromosome-sorter",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "chromosome-sort=main:main",
        ],
    },
    keywords="chromosome karyotype sorting machine-learning bioinformatics image-processing",
)