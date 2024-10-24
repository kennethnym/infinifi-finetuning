import audiocraft.models
from audiocraft.utils import export
from audiocraft import train
from audiocraft.data.audio import audio_write

signature = 'aec31258'

xp = train.main.get_xp_from_sig(signature)
export.export_lm(xp.folder / 'checkpoint.th', '/checkpoints/infinifi/state_dict.bin')

## Case 2) you used a pretrained model. Give the name you used without the //pretrained/ prefix.
## This will actually not dump the actual model, simply a pointer to the right model to download.
export.export_pretrained_compression_model('facebook/encodec_32khz', '/checkpoints/infinifi/compression_state_dict.bin')

model = audiocraft.models.MusicGen.get_pretrained('/checkpoints/infinifi/')

model.set_generation_params(duration=60)
descriptions = [
    "soothing lo-fi beat featuring gentle, melodic guitar riffs.",
    "soothing lo-fi beat featuring gentle, melodic guitar riffs.",
    "soothing lo-fi beat featuring gentle, melodic guitar riffs.",
    "soothing lo-fi beat featuring gentle, melodic guitar riffs.",
    "soothing lo-fi beat featuring gentle, melodic guitar riffs.",
]
wav = model.generate(descriptions)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)


