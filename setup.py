from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="newsrag", version="0.2.0",
    description="Auditable, provider-neutral RAG for newsroom research",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Shehata El-sayed", license="Apache-2.0",
    classifiers=["Programming Language :: Python :: 3"],
    packages=find_packages("src"),
    package_dir={"": "src"}, python_requires=">=3.10",
    install_requires=[], extras_require={"pdf": ["pypdf>=5,<7"]},
)
