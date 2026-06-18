from os import environ

from huggingface_hub import HfApi

HF_TOKEN = environ["HF_TOKEN"]
REPO_ID = "duyle2408/repo_name"
FOLDER_PATH = "/path/to/your/folder"

api = HfApi(token=HF_TOKEN)

api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",   # hoặc "model", "space"
    private=False,
    exist_ok=True,
)

api.upload_folder(
    folder_path=FOLDER_PATH,
    repo_id=REPO_ID,
    repo_type="dataset",
)