import re

def remove_emojis(text):
    # This regex matches a wide range of emojis and miscellaneous symbols
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002300-\U000023FF"  # Miscellaneous Technical
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    # Manual fixes for gender symbols often used
    text = text.replace('\u2642', 'M')
    text = text.replace('\u2640', 'F')
    # Remove any stray non-ascii characters if they might cause issues
    # text = text.encode('ascii', 'ignore').decode('ascii')
    return text

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

cleaned_content = remove_emojis(content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print("Emoji removal complete.")
