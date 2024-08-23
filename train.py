import os
import datasets
from huggingface_hub import hf_hub_download

from torch.utils.data import DataLoader

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SAMPLE_RATE = 44100
REPO = "vikhyatk/lofi"

auth_token = os.environ["TOKEN"]

ds = datasets.load_dataset(
    REPO,
    use_auth_token=auth_token,
    split="train",
    streaming=True,
)

loader = DataLoader(ds, batch_size=32, num_workers=4)

i = 0
for row in iter(loader):
    if i == 1:
        break

    paths = row["audio"]["path"]
    for path in paths:
        hf_hub_download(REPO, filename=path, local_dir="cache", token=auth_token)

    i += 1
