# Определяет эмоции по тексту
import torch
from aniemore.recognizers.text import TextRecognizer
from aniemore.models import HuggingFaceModel

model = HuggingFaceModel.Text.Bert_Tiny2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tr = TextRecognizer(model=model, device=device)

result = tr.recognize('ура, это работает!!!', return_single_label=True)
print("Эмоция:", result)
