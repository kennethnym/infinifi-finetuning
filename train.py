import os
import datasets
import torchaudio

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
ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE))

loader = DataLoader(ds, batch_size=32, num_workers=4)

i = 0
for row in iter(loader):
    if i == 1:
        break

    audio_list = row["audio"]["array"]
    for i, data in enumerate(audio_list):
        torchaudio.save(f"cache/{i}.mp3", data, sample_rate=SAMPLE_RATE, format="mp3")

    i += 1
