import os
import datasets
import random
import json

from torch.utils.data import DataLoader
from keybert import KeyBERT

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SAMPLE_RATE = 44100
TRAIN_SIZE = 0.8
REPO = "vikhyatk/lofi"

batch_count = int(os.environ["BATCH_COUNT"])

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
    split="train",
    streaming=True,
)
ds = ds.shuffle()
ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE, decode=False))

loader = DataLoader(ds, batch_size=32)

kb_model = KeyBERT()

# ignore any row with prompt that contains these words because they aren't very good.
ignore_words = set(['cello', 'funky'])

instruments = ['piano', 'guitar', 'violin', 'flute', 'xylophone']
all_moods = ['nostalgic', 'chill', 'chilling', 'uplift', 'dreamy', 'exhausted', 'intimate', 'dramatic', 'frustrated', 'uplifting', 'soulful', 'calm', 'zen', 'lively', 'cozy', 'peaceful', 'sensual', 'playful', 'joyous', 'passionate', 'enigmatic', 'soothing']

i = 0
for row in iter(loader):
    print(f"processing batch {i}...")

    if i == batch_count:
        break

    keywords = kb_model.extract_keywords(row["prompt"])

    audio_list = row["audio"]["bytes"]
    for j, data in enumerate(audio_list):
        music_id = row["id"][j]
        prompt = row["prompt"][j]

        with open(f"audiocraft/dataset/lofi/{music_id}.mp3", "wb") as f:
            f.write(data)

        instrument = None
        for name in instruments:
            if name in prompt.lower():
                instrument = name
                break
        moods = [m for m in all_moods if m in prompt.lower()]

        entry = {
            "sample_rate": SAMPLE_RATE,
            "file_extension": "mp3",
            "description": prompt,
            "keywords": ", ".join([k[0] for k in keywords[j][:2]]),
            "genre": "lofi",
            "duration": 29,
            "path": f"dataset/lofi/{music_id}.mp3",
        }
        if instrument:
            entry["instrument"] = instrument
        if len(moods) > 0:
            entry["moods"] = moods

        if random.random() < TRAIN_SIZE:
            train_manifest_file.write(json.dumps(entry) + "\n")
        else:
            eval_manifest_file.write(json.dumps(entry) + "\n")

    i += 1

train_manifest_file.close()
eval_manifest_file.close()
