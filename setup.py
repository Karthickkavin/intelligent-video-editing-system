"""
Setup configuration for the Intelligent Video Editing System.
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    install_requires = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="intelligent-video-editing-system",
    version="1.0.0",
    description="AI-powered video editing system with intelligent automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Karthickkavin",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "video-editor=examples.example_usage:main",
        ]
    },
)
