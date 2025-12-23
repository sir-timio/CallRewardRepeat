import os
import shutil
from typing import Any, Dict, List

import pandas as pd
from clearml import Dataset, InputModel, Logger, OutputModel, Task

PROJECT_NAME = "DST"
REQUIREMENTS_PATH = "requirements.txt"
DEFAULT_DOCKER_IMAGE = None
OUTPUT_URI = None
HF_TOKEN=os.getenv("HF_TOKEN")
DOCKER_ARGS=None
DATASET_ID = None

def copy_data(dataset_id: str=None):
    dataset_id = dataset_id or DATASET_ID
    if dataset_id:
        clearml_data_path = Dataset.get(dataset_id=dataset_id).get_local_copy()
        shutil.copytree(clearml_data_path, "./data", dirs_exist_ok=True)

def update_data_path(data_path: str, dataset_id: str = None):
    dataset_id = dataset_id or DATASET_ID
    if dataset_id:
        clearml_data_path = Dataset.get(dataset_id).get_local_copy()
        data_path = data_path.replace("./data", clearml_data_path)
    return data_path

def create_task(
    task_name: str,
    project_name: str = None,
    output_uri: str = None,
) -> Task:
    """Creates a clearML task

    Args:
        task_name (str): Task name
        instance (str, optional): ClearML instance. Defaults to "mlplat"
        project_name (str, optional): ClearML project name. Defaults to "AIKTK"
        output_uri (str, optional): S3 URI for output. Defaults to None

    Returns:
        Task: ClearML task instance
    """
    # Define task output URI
    output_uri = output_uri or OUTPUT_URI
    project_name = project_name or PROJECT_NAME
    # Create task instance
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        output_uri=output_uri,
        task_type="training",
        reuse_last_task_id=False,
    )
    return task


def report_df(df: pd.DataFrame, name: str = "Results", series: str = None):
    """Reports a pandas DataFrame as a table in clearML

    Args:
        df (pd.DataFrame): DataFrame to report
        name (str, optional): Name of the table. Defaults to "Results"
        series (str, optional): Series name. Defaults to None
    """
    if series is None:
        series = name
    Logger.current_logger().report_table(
        name, series=series, iteration=0, table_plot=df
    )


def upload_artifact(path: str, name: str = None):
    """Uploads an artifact to clearML

    Args:
        path (str): Path to the artifact
        name (str, optional): Name of the artifact. Defaults to None
    """
    MODEL_FORMATS = [".pt", ".bin", ".safetensors"]

    # If the artifact is a model, use the OutputModel to upload it
    if any(path.endswith(fmt) for fmt in MODEL_FORMATS):
        kwargs = {"name": name} if name else {}
        out_model = OutputModel(**kwargs)
        out_model.update_weights(path, auto_delete_file=False)
    # Otherwise, use upload_artifact to upload it
    else:
        name = path.split("/")[-1] if name is None else name
        Task.current_task().upload_artifact(artifact_object=path, name=name)


def execute_remotely(
    task: Task,
    queue: str,
    docker_image: str = None,
    env_dict: Dict[str, Any] = {},
    reqs: str = REQUIREMENTS_PATH,
):
    """Executes a clearML task remotely

    Args:
        task (Task): ClearML task instance
        queue (str): Queue to execute the task on
        docker_image (str, optional): Docker image to use
        env_dict (Dict[str, Any], optional): Environment variables to set
        reqs (str, optional): Path to requirements file
    """
    if docker_image is None:
        docker_image = DEFAULT_DOCKER_IMAGE
    env_dict = {"HF_TOKEN": HF_TOKEN, **env_dict}
    env_args = " ".join([f"--env {k}={v}" for k, v in env_dict.items()])
    task.set_base_docker(
        docker_image=docker_image, docker_arguments=f"{DOCKER_ARGS}\n{env_args}",
    )
    if reqs is not None:
        task.set_packages(packages=reqs)
    task.execute_remotely(queue_name=queue)


def download_model(model_id: str) -> str:
    """Downloads an artifact from clearML using a model ID

    Args:
        model_id (str): Model ID to download the artifact from

    Returns:
        str: Path to the downloaded artifact
    """
    input_model = InputModel(model_id=model_id)
    try:
        # Try to get the weights package (and unzip it at once)
        path = input_model.get_weights_package(return_path=True)
    except ValueError as e:
        print(e)
        # If it fails, try to get the weights directly
        path = input_model.get_weights()
    return path


def download_dataset_id(id: str) -> str:
    """Downloads a dataset from clearML using a dataset ID

    Args:
        id (str): Dataset ID to download
    Returns:
        str: Path to the downloaded dataset
    """
    dataset_class = Dataset.get(dataset_id=id)
    return dataset_class.get_local_copy()
