# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from setuptools import setup

pkg_dir = Path(__file__).parent.absolute()

with open(pkg_dir / "requirements.txt") as fh:
    requirements = []
    for line in fh:
        line = line.strip()
        if "#" in line:
            line = line[:line.index("#")].strip()
        if len(line) == 0:
            continue
        requirements.append(line)

setup(
    name="stf",
    version="1.5.0",
    packages=["stf"],
    description="Short term forcasting",
    url="https://github.com/alliander-opensource/short-term-forecasting",
    author="Alliander N.V",
    author_email="korte.termijn.prognoses@alliander.com",
    license="MPL-2.0",
    keywords=['energy', 'forecasting', 'machinelearning'],
    include_package_data=True,
    python_requires=">=3.9.0",
    install_requires=requirements,
    setup_requires=[
        "wheel~=0.36.2", "Cython~=0.29.21"
    ],
    tests_require=[
        "pytest", "pytest-cov"
    ],
    classifiers=[
        "Development Status :: 5 - Production",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MPL-2.0",
        "Programming Language :: Python :: 3.9",
    ],
)
