# type: ignore

import os
from setuptools import setup

BASE_PATH = os.path.abspath(os.path.dirname(__file__))


setup(
    name="optuna_memorydump",
    version="0.0.1",
    author="Masashi Shibata",
    author_email="shibata_masashi@cyberagent.co.jp",
    url="https://github.com/c-bata/optuna-memorydump",
    description="",
    long_description=open(os.path.join(BASE_PATH, "README.md")).read(),
    long_description_content_type="text/markdown",
    classifiers=[],
    py_modules=['optuna_memorydump'],
    install_requires=["optuna"],
    extras_require={},
    tests_require=[],
    keywords="optuna",
    license="MIT License",
    include_package_data=True,
    test_suite="",
)
