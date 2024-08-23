import os
from datasets import Audio, Dataset, load_dataset
from random import randint

SAMPLE_RATE = 44100

os.makedirs("./audiocraft/dataset/lofi-train", exist_ok=True)

ds = load_dataset(
    "vikhyatk/lofi",
    split="train[:100]",
)
ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
