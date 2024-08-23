import os
import datasets

from torch.utils.data import DataLoader

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SAMPLE_RATE = 44100

ds = datasets.load_dataset(
    "vikhyatk/lofi",
    use_auth_token=os.environ["TOKEN"],
    split="train",
    num_proc=64,
    streaming=True,
)

loader = DataLoader(ds, batch_size=32, num_workers=4)

for row in iter(ds):
    print(row)
