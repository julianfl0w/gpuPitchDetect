#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="gpuPitchDetect",
    version="1.0",
    description="Runs a GLSL Vulkan compute shader on GPU",
    author="Julian Loiacono",
    author_email="jcloiacon@gmail.com",
    url="https://github.com/julianfl0w/gpuPitchDetect",
    packages=find_packages(),
    package_data={
        # everything
        "": ["*"]
    },
    include_package_data=True,
)
