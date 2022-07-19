from setuptools import setup
import os.path
import codecs


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="uci_datasets",
    version=get_version(os.path.join("uci_datasets", "__init__.py")),
    description="UCI regression datasets.",
    url="https://github.com/treforevans/uci_datasets",
    author="Trefor Evans",
    author_email="trefor.evans@mail.utoronto.ca",
    license="MIT",
    packages=["uci_datasets"],
    package_data={"uci_datasets": ["**/*.csv.gz"]},
    install_requires=["numpy"],
    zip_safe=True,
)
