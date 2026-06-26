import os
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path='src',
    repo_id='Schiro/Monitoraggio_Della_Reputazione_Online',
    repo_type='space',
    token=os.environ['HF_TOKEN']
)
print('Deploy completed!')