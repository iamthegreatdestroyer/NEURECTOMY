from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="neurectomy-sdk",
    version="1.0.0",
    author="Neurectomy Team",
    author_email="info@neurectomy.ai",
    description="Production-ready Python SDK for Neurectomy API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iamthegreatdestroyer/NEURECTOMY",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "urllib3>=1.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=23.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
    },
    keywords=[
        "neurectomy",
        "ai",
        "llm",
        "compression",
        "sdk",
    ],
)
