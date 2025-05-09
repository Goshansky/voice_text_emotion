# На основе аудио угадывает эмоцию
import torch
from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.models import HuggingFaceModel

model = HuggingFaceModel.Voice.WavLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vr = VoiceRecognizer(model=model, device=device)
result = vr.recognize('audio.mp3', return_single_label=True)
print("Эмоция:", result)
