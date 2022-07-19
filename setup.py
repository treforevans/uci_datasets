from setuptools import setup

setup(
    name="uci_datasets",
    version="1.1.0",
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
