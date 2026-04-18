INSTRUCTIONS = {
    "default": "Ответь используя только данный контекст.",
    "detail":  "Дай детальный ответ только из этих документов.",
}

def pick_instruction(category: str = "default") -> str:
    return INSTRUCTIONS.get(category, INSTRUCTIONS["default"])
