# Переводит речь в текст
import whisper

model = whisper.load_model("small")  # можно выбрать tiny, base, small, medium, large
result = model.transcribe("audio.mp3")  # поддерживаются также .wav, .m4a, .webm

print(result["text"])
