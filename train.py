import os
from datasets.arrow_reader import DownloadConfig
import huggingface_hub
import datasets
from random import randint

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SAMPLE_RATE = 44100

ds = datasets.load_dataset(
    "vikhyatk/lofi",
    split="train[10:20]",
    download_config=DownloadConfig(num_proc=32),
    num_proc=64,
)
