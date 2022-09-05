import pickle
import mlflow
import shutil
from pathlib import Path


# from .bis_mlflow_sklearn_wrapper import SKLearnWrapper
# from .bis_mlflow_sklearn_wrapper import artifacts
import bis_mlflow_model_wrappers

mlflow_custom_sklearn_model_path = "prod_model"


def add_package_to_environment_file():
    conda_env_file = Path(f"{mlflow_custom_sklearn_model_path}/conda.yaml")
    if conda_env_file.is_file():
        with open(conda_env_file, "r") as f:
            contents = f.readlines()

        contents.insert(
            contents.index("- pip:\n") + 1,
            "  - git+ssh://git@github.com/dhruv-108/bis-mlflow-model-wrappers.git@main\n",
        )
        with open(conda_env_file, "w") as f:
            contents = "".join(contents)
            f.write(contents)
    else:
        print(f"{conda_env_file} does not exist!")


def save_model(model) -> None:

    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    # # Model registry does not work with file store
    # if tracking_url_type_store != "file":

    prod_model_dir = Path("prod_model")

    if prod_model_dir.exists() and prod_model_dir.is_dir():
        shutil.rmtree(prod_model_dir)

    # Register the model
    # There are other ways to use the Model Registry, which depends on the use case,
    # please refer to the doc for more information:
    # https://mlflow.org/docs/latest/model-registry.html#api-workflow

    pickle.dump(model, open("model.pkl", "wb"))

    mlflow.pyfunc.save_model(
        path=mlflow_custom_sklearn_model_path,
        python_model=bis_mlflow_model_wrappers.SKLearnWrapper(),
        artifacts=bis_mlflow_model_wrappers.sklearn_artifacts,
    )

    add_package_to_environment_file()

    mlflow.pyfunc.log_model(
        artifact_path=mlflow_custom_sklearn_model_path,
        loader_module=None,
        data_path=None,
        conda_env=f"{mlflow_custom_sklearn_model_path}/conda.yaml",
        python_model=bis_mlflow_model_wrappers.SKLearnWrapper(),
        registered_model_name="Custom_mlflow_model",
        artifacts=bis_mlflow_model_wrappers.sklearn_artifacts,
    )

    mlflow.log_artifact("KNIME Data Ingest.zip")
    mlflow.log_artifact("Features.json")
    mlflow.log_artifact("data_transform.py")
