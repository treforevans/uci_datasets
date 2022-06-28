from setuptools import setup

setup(
    name="uci_datasets",
    version="1.0",
    description="UCI regression datasets.",
    url="https://github.com/treforevans/uci_datasets",
    author="Trefor W. Evans",
    author_email="trefor.evans@mail.utoronto.ca",
    packages=["uci_datasets"],
    package_data={'uci_datasets': ["**/*.csv.gz"]},
    zip_safe=True,
)
