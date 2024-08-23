import os
import datasets
import pydub
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
ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE))

loader = DataLoader(ds, batch_size=32, num_workers=4)

i = 0
for row in iter(loader):
    if i == 1:
        break

    audio_list = row["audio"]["array"]
    for i, data in enumerate(audio_list):
        audio_segment = pydub.AudioSegment(
            data.to_bytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=data.dtype.itemsize,
            channels=1,
        )
        audio_segment.export(f"cache/{i}.mp3", format="mp3")

    i += 1
