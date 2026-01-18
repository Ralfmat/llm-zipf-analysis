import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_KEY")
client = Groq(api_key=API_KEY)
model_name = "llama-3.3-70b-versatile"
# temperatures = [0.2, 0.7, 1.5]
temperatures = [1.5]

PROMPT_TEMPLATE = """Write a comprehensive, detailed essay in English about the following topic: {topic}.

Please adhere to the following guidelines:

Length: The essay must be at least 800 words long.

Style: Write in a formal, academic, yet engaging tone, similar to an article in 'The Atlantic' or 'The Economist'.

Structure: Include a clear introduction, multiple body paragraphs exploring different angles of the topic, and a strong conclusion.

Content: Avoid bullet points or lists; rely on full, descriptive sentences and coherent paragraphs.

Do not mention that you are an AI. Just provide the text of the essay."""

# --- LISTA TEMATÓW ---
# Wybrałem tematy szerokie i abstrakcyjne, żeby zmusić AI do "głębokiego" pisania.
topics = [
    ## LIMIT OSIĄGNIĘTY, RESZTA W INNYM PLIKU
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

print(f"--- GENERATOR ESEJÓW (Zipf's Law Data Mining) ---")
print(f"Model: {model_name}")
print(f"Szacowana długość jednego pliku: ~{len(topics) * 800} słów")

for temp in temperatures:
    filename = f"ai_text2_2_temp_{temp}.txt"
    print(f"\nGenerowanie pliku: {filename} (Temperatura: {temp})...")

    with open(filename, "w", encoding="utf-8") as f:
        for i, topic in enumerate(topics):
            final_prompt = PROMPT_TEMPLATE.format(topic=topic)

            print(f"  [{i + 1}/{len(topics)}] Piszę esej o: {topic[:40]}...", end="\r")

            try:
                start_time = time.time()

                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": final_prompt}],
                    model=model_name,
                    temperature=temp,
                    max_tokens=6000,
                )

                content = response.choices[0].message.content

                # Zapis do pliku
                f.write(f"--- TOPIC: {topic} ---\n")
                f.write(content + "\n\n")

                elapsed = time.time() - start_time

                time.sleep(3)

            except Exception as e:
                print(f"\n  [!] Błąd przy temacie '{topic}': {e}")
                time.sleep(5)

    print(f"\nUKOŃCZONO: {filename}")
