import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_KEY")
client = genai.Client(api_key=API_KEY)

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


MODEL_NAME = "gemini-2.5-flash"

temperatures = [1.5]
# temperatures = [0.2, 0.7, 1.5]


PROMPT_ESSAY = """Write a formal, academic essay about: {topic}.
Rules:
1. Tone: Professional, objective, factual.
2. Vocabulary: Standard academic English.
3. Structure: Introduction, body paragraphs, conclusion.
4. Length: Approx 800 words."""

PROMPT_CREATIVE = """Write a highly experimental, stream-of-consciousness literary piece about: {topic}.
Rules:
1. Tone: Abstract, poetic, weird.
2. Vocabulary: Use extremely rare, archaic, and obscure words.
3. Style: Avoid clichés. Be unpredictable.
4. Length: Approx 800 words."""


topics = [
    "The impact of Artificial Intelligence on the future of creative professions",
    "The psychological effects of social media on modern society",
    "The ethical implications of genetic engineering in humans",
    "The role of cryptocurrency in the future global economy",
    "The history and cultural significance of tea ceremonies around the world",
    "The paradox of choice: Why having more options makes us less happy",
    "The influence of ancient Greek philosophy on modern western thought",
    "The potential consequences of discovering extraterrestrial life",
    "The evolution of storytelling from oral traditions to digital media",
    "The importance of biodiversity for the survival of the human race",
    "The concept of time perception: Why time seems to speed up as we age",
    "The architectural challenges of building colonies on Mars",
    "The relationship between music and memory retention",
    "The socio-economic impact of the Industrial Revolution",
    "The future of privacy in the age of big data surveillance",
    "The complete history of the Roman Empire",
    "Global warming: causes, effects, and future solutions",
    "The rapid development of medicine and surgery during World War II",
    "The history and cultural significance of Italian cuisine",
    "The Apollo program and the moon landing missions",
    "The core concepts and history of Nihilistic philosophy",
    "Art Nouveau (Secession) in architecture and design",
    "The unique biodiversity and evolution of the Galapagos Islands",
    "The causes and global consequences of the 2007 financial crisis"
]

print(f"GENERATOR GEMINI")
print(f"Model: {MODEL_NAME}")

for temp in temperatures:
    filename = f"gemini_temp_{temp}.txt"
    print(f"\nGenerowanie pliku: {filename} (Temp: {temp})...")

    with open(filename, "w", encoding="utf-8") as f:

        generate_config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=8000,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                )
            ]
        )


        print(f"  > Generowanie esejów...")
        for topic in topics:
            prompt = PROMPT_ESSAY.format(topic=topic)
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=generate_config
                )

                f.write(f"--- TYPE: ESSAY | TOPIC: {topic} ---\n")
                f.write(response.text + "\n\n")
                print(f"    [OK] Esej: {topic[:20]}...", end="\r")
                time.sleep(4)
            except Exception as e:
                print(f"\n    [!] Błąd: {e}")
                time.sleep(5)


        # print(f"\n  > Generowanie tekstów kreatywnych...")
        # for topic in topics:
        #     prompt = PROMPT_CREATIVE.format(topic=topic)
        #     try:
        #         response = client.models.generate_content(
        #             model=MODEL_NAME,
        #             contents=prompt,
        #             config=generate_config
        #         )
        #
        #         f.write(f"--- TYPE: CREATIVE | TOPIC: {topic} ---\n")
        #         f.write(response.text + "\n\n")
        #         print(f"    [OK] Creative: {topic[:20]}...", end="\r")
        #         time.sleep(4)
        #     except Exception as e:
        #         print(f"\n    [!] Błąd: {e}")
        #         time.sleep(5)

    print(f"\nUKOŃCZONO: {filename}")

print("\n--- GOTOWE ---")