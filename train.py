import os
import datasets
from huggingface_hub import hf_hub_download

from torch.utils.data import DataLoader

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SAMPLE_RATE = 44100
REPO = "vikhyatk/lofi"

ds = datasets.load_dataset(
    REPO,
    use_auth_token=os.environ["TOKEN"],
    split="train",
    streaming=True,
)

loader = DataLoader(ds, batch_size=32, num_workers=4)

i = 0
for row in iter(loader):
    if i == 1:
        break

    relpath = row["audio"]["path"]
    hf_hub_download(REPO, filename=relpath, local_dir="cache")
