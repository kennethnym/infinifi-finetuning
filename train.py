import os
import datasets
import random
import json

from torch.utils.data import DataLoader

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SAMPLE_RATE = 44100
SAMPLE_SIZE = 200
TRAIN_SIZE = 0.8
REPO = "vikhyatk/lofi"

auth_token = os.environ["TOKEN"]

os.makedirs("audiocraft/dataset/lofi", exist_ok=True)
os.makedirs("audiocraft/egs/train", exist_ok=True)
os.makedirs("audiocraft/egs/eval", exist_ok=True)

with open("audiocraft/config/dset/audio/lofi.yaml", "w") as f:
    content = f"""# @package __global__

datasource:
    max_sample_rate: {SAMPLE_RATE}
    max_channels: 1

    train: egs/train
    valid: egs/eval
    evaluate: egs/eval
    generate: egs/train"""
    f.write(content)

train_manifest_file = open("audiocraft/egs/train/data.jsonl", "w")
eval_manifest_file = open("audiocraft/egs/eval/data.jsonl", "w")

ds = datasets.load_dataset(
    REPO,
    use_auth_token=auth_token,
    split="train",
    streaming=True,
)
ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE, decode=False))

loader = DataLoader(ds, batch_size=32, num_workers=4)

i = 0
for row in iter(loader):
    if i == SAMPLE_SIZE:
        break

    audio_list = row["audio"]["bytes"]
    for j, data in enumerate(audio_list):
        music_id = row["id"][j]

        with open(f"audiocraft/dataset/lofi/{music_id}.mp3", "wb") as f:
            f.write(data)

        entry = {
            "sample_rate": SAMPLE_RATE,
            "file_extension": "mp3",
            "description": row["prompt"][j],
            "duration": 29,
            "path": f"dataset/lofi/{music_id}.mp3",
        }

        if random.random() < TRAIN_SIZE:
            train_manifest_file.write(json.dumps(entry) + "\n")
        else:
            eval_manifest_file.write(json.dumps(entry) + "\n")

    i += 1

train_manifest_file.close()
eval_manifest_file.close()
