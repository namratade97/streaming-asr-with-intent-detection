import re

def extract_intent_from_supervision(text: str) -> str:
    text = text.strip()
    match = re.search(r'(<[^>]+>)$', text)
    if match:
        return match.group(1)
    else:
        return None

if __name__ == '__main__':
    sample_texts = [
        "EVENT REMINDER MONA TUESDAY <alarm_set>",
        "PUT MEETING WITH PAWEL FOR TOMORROW TEN AM <calendar_set>",
        "NO INTENT HERE"
    ]
    for text in sample_texts:
        intent = extract_intent_from_supervision(text)
        print(f"Text: {text} -> Intent: {intent}")
