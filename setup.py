from setuptools import find_packages, setup

setup(
    name="bis_mlflow_model_wrappers",
    packages=find_packages(include=["bis_mlflow_model_wrappers"]),
    version="0.1.10",
    description="WITS BIS Mlflow model wrappers",
    author="Dhruv Bhugwan",
    license="MIT",
    install_requires=["mlflow", "numpy", "mlflow", "pathlib"],
)
