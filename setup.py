import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Read requirements
reqs = (HERE / "requirements.txt").read_text()

setup(
    name="ephys_gpt",
    python_requires=">=3.13, <3.14",
    install_requires=reqs,
    packages=find_packages(),
)
