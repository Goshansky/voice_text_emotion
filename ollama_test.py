from ollama import chat, ChatResponse

# история сообщений (включая system, user, assistant)
messages = [
    {
        "role": "system",
        "content": (
            "Ты — доброжелательный психолог-консультант, обученный поддерживать "
            "пациентов в трудные моменты. Ты говоришь мягко, деликатно, поддерживающе. "
            "Ты не даёшь медицинские советы, но стараешься понять чувства человека и помочь ему выразить эмоции."
        )
    },
    {
        "role": "user",
        "content": "Я хочу умереть"
    }
]

response: ChatResponse = chat(
    model='gemma3',
    messages=messages
)

print(response.message.content)

# Добавляем ответ в историю
messages.append({
    "role": "assistant",
    "content": response.message.content
})
