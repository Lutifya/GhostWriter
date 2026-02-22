"""Package setup for GhostWriter."""

from setuptools import find_packages, setup

setup(
    name="ghostwriter",
    version="0.1.0",
    description="GhostWriter – System-Wide Live Captions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "faster-whisper>=0.9.0",
        "numpy>=1.24.0",
        "sounddevice>=0.4.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-mock>=3.10",
        ]
    },
    entry_points={
        "console_scripts": [
            "ghostwriter=ghostwriter.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
