import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_KEY")

client = Groq(api_key=API_KEY)
model_name = "llama-3.3-70b-versatile"
temperatures = [0.2, 0.7, 1.5]

# --- LISTA TEMATÓW (Żeby zmusić AI do użycia różnego słownictwa) ---
# Zmieniliśmy polecenia na "Write a long...", żeby uzyskać więcej tekstu.
prompts = [
    "Write a long detailed essay about the history of the internet.",
    "Describe a fantasy world where islands float in the sky. Be descriptive.",
    "Explain quantum physics using cooking metaphors in a long paragraph.",
    "Write a long horror story set in an abandoned library.",
    "Describe the biological process of photosynthesis in detail.",
    "Write a long philosophical debate between a robot and a human about feelings.",
    "Describe the hustle and bustle of a market in Tokyo.",
    "Write a long letter from an explorer discovering a new continent.",
    "Explain the economic causes of the Great Depression.",
    "Write a long review of a fictional movie that doesn't exist.",
    "Describe the taste, smell, and texture of your favorite meal.",
    "Write a long guide on how to survive a zombie apocalypse.",
    "Explain how a car engine works to someone who knows nothing about cars.",
    "Write a long poem about the loneliness of a lighthouse keeper.",
    "Describe a long futuristic city made entirely of glass.",
    "Write a long dialogue between two cats planning to take over the world.",
    "Explain the concept of 'entropy' in thermodynamics.",
    "Describe the feeling of waking up on a cold winter morning.",
    "Write a long myth about how the stars were created.",
    "Write a long technical report about a fictional new invention."
]

print(f"--- ROZPOCZYNAM GENEROWANIE DANYCH (Model: {model_name}) ---")

for temp in temperatures:
    filename = f"ai_text_temp_{temp}.txt"
    print(f"\nGenerowanie pliku: {filename} (Temperatura: {temp})...")

    # Otwieramy plik w trybie zapisu ('w'), co wyczyści stary plik jeśli istnieje
    with open(filename, "w", encoding="utf-8") as f:

        for i, prompt in enumerate(prompts):
            print(f"  [{i + 1}/{len(prompts)}] Piszę o: {prompt[:30]}...", end="\r")

            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                    temperature=temp,
                    max_tokens=1024,  # Pozwalamy na długie odpowiedzi
                )

                content = response.choices[0].message.content

                # Zapisujemy do pliku
                f.write(content + "\n\n")

                # Ważne: Pauza, żeby Groq nie zablokował nas za spamowanie (Rate Limit)
                time.sleep(2)

            except Exception as e:
                print(f"\n  Błąd przy: {prompt[:20]} -> {e}")
                time.sleep(5)  # Dłuższa przerwa jak wystąpi błąd

    print(f"\nUKOŃCZONO: {filename} - Zapisano na dysku.")
