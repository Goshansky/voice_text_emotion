# Сначала переводит голос в текст, а потом угадывает эмоцию точнее
# Отличается от emotion_multi тем, что текст можно вставить свой
import torch
from aniemore.recognizers.multimodal import VoiceTextRecognizer
from aniemore.utils.speech2text import SmallSpeech2Text
from aniemore.models import HuggingFaceModel

# Загрузка модели
model = HuggingFaceModel.MultiModal.WavLMBertFusion
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Распознавание речи из аудио
s2t_model = SmallSpeech2Text()
text = s2t_model.recognize('audio.mp3').text
print("Распознанный текст:", text)

# Распознавание эмоций по аудио и тексту
vtr = VoiceTextRecognizer(model=model, device=device)
result = vtr.recognize(('audio.mp3', text), return_single_label=True)

print("Эмоция:", result)

