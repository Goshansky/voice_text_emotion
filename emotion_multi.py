# Сначала переводит голос в текст, а потом угадывает эмоцию точнее
# Отличается от emotion_voice+ тем, что переведенный текст сразу кидает на анализ
import torch
from aniemore.recognizers.multimodal import MultiModalRecognizer
from aniemore.utils.speech2text import SmallSpeech2Text
from aniemore.models import HuggingFaceModel

model = HuggingFaceModel.MultiModal.WavLMBertFusion
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Инициализация речепреобразователя и мульти-модального распознавателя
mr = MultiModalRecognizer(model=model, s2t_model=SmallSpeech2Text(), device=device)

# Получение результата распознавания
result = mr.recognize('audio.mp3', return_single_label=True)

# Вывод результата
print("Распознанная эмоция:", result)
