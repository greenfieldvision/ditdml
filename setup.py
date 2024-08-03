import setuptools

NAME = "ditdml"
VERSION = "0.0.9"

INSTALL_REQUIRES = [
    "more-itertools",
    "numpy",
    "scipy",
    "scikit-learn",
    "Pillow",
    "psiz==0.5.1",
]

setuptools.setup(
    name=NAME,
    version=VERSION,
    description="Data Interfaces for Triplet-based Distance Metric Learning.",
    url="https://github.com/greenfieldvision/ditdml",
    author="Radu Dondera",
    license="MIT License",
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
)
