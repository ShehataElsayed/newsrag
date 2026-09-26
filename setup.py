from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="newsrag", version="0.3.1",
    description="Auditable, provider-neutral RAG for newsroom research",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Shehata El-sayed", license="Apache-2.0",
    url="https://github.com/ShehataElsayed/newsrag",
    project_urls={"Source": "https://github.com/ShehataElsayed/newsrag",
                  "Issues": "https://github.com/ShehataElsayed/newsrag/issues",
                  "Documentation": "https://github.com/ShehataElsayed/newsrag/tree/main/docs"},
    keywords="rag journalism research citations arabic",
    classifiers=["Development Status :: 3 - Alpha",
                 "Intended Audience :: Developers",
                 "License :: OSI Approved :: Apache Software License",
                 "Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.10",
                 "Programming Language :: Python :: 3.11",
                 "Programming Language :: Python :: 3.12",
                 "Programming Language :: Python :: 3.13",
                 "Topic :: Text Processing :: Indexing"],
    packages=find_packages("src"),
    package_dir={"": "src"}, python_requires=">=3.10",
    install_requires=[], extras_require={"pdf": ["pypdf>=5,<7"],
                                      "multilingual": ["sentence-transformers>=3,<6"],
                                      "arabic": ["snowballstemmer>=2.2,<4"],
                                      "web": ["trafilatura>=1.12,<3"]},
)
