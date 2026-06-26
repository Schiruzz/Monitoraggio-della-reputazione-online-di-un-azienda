import os
from huggingface_hub import HfApi

api = HfApi()
token = os.environ['HF_TOKEN']
repo_id = 'Schiro/Monitoraggio_Della_Reputazione_Online'

# Upload src files
api.upload_folder(
    folder_path='src',
    repo_id=repo_id,
    repo_type='space',
    token=token
)

# Upload requirements.txt
api.upload_file(
    path_or_fileobj='requirements.txt',
    path_in_repo='requirements.txt',
    repo_id=repo_id,
    repo_type='space',
    token=token
)

print('Deploy completed!')