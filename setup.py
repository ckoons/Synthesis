#!/usr/bin/env python3
"""
Setup script for Synthesis - Execution Engine for Tekton
"""

from setuptools import setup, find_packages

setup(
    name="synthesis",
    version="0.1.0",
    description="Execution engine for the Tekton project ecosystem",
    author="Tekton Team",
    author_email="info@tekton.ai",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.8.0",
        "pydantic>=1.9.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "numpy>=1.20.0",
        "asyncio>=3.4.3",
        "tekton-core>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.16.0",
            "black>=21.10b0",
            "isort>=5.9.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "synthesis=synthesis.cli:main",
        ],
    },
    python_requires=">=3.9",
)