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


MODEL_NAME = "gemini-2.5-pro"

temperatures = [1.5]
# temperatures = [0.2, 1.0, 1.5]


PROMPT_ESSAY = """Write a formal, academic essay about: {topic}.
Rules:
1. Tone: Professional, objective, factual.
2. Vocabulary: Standard academic English.
3. Structure: Introduction, body paragraphs, conclusion.
4. Length: Approx 800 words."""

PROMPT_WIKI = """Write a comprehensive, detailed encyclopedic article about: {topic}.
Rules:
1. Tone: Neutral, informative, objective (Wikipedia style).
2. Vocabulary: Precise, descriptive, high-density of facts.
3. Structure: Definition, History/Origins, Characteristics, Significance.
4. Length: Approx 1000 words."""

topics_wiki = [
    "The rise and fall of the Ottoman Empire",
    "The feudal system in medieval Europe",
    "The construction and significance of the Great Wall of China",
    "The history of the Maya civilization and their astronomical achievements",
    "The French Revolution: causes, key events, and consequences",
    "The Golden Age of Piracy in the Caribbean",
    "The Vikings: culture, exploration, and warfare",
    "The Manhattan Project and the development of atomic weapons",
    "The history of the Silk Road and its impact on trade",
    "The partition of India and Pakistan in 1947",

    "The theory of general relativity and its impact on physics",
    "The biological process of photosynthesis in plants",
    "The structure and function of the human DNA molecule",
    "The history of the internet: from ARPANET to the World Wide Web",
    "The principles of quantum mechanics and wave-particle duality",
    "The geological theory of plate tectonics",
    "The evolution of the internal combustion engine",
    "Black holes: formation, types, and detection",
    "The discovery and medical application of antibiotics",
    "The engineering challenges of the Panama Canal",

    "The ecosystem and biodiversity of the Amazon Rainforest",
    "The formation and geological features of the Grand Canyon",
    "The climate and geography of Antarctica",
    "The Great Barrier Reef: marine life and environmental threats",
    "The meteorological phenomenon of El Niño",
    "The physiography of the Himalayas",
    "The Sahara Desert: climate, history, and ecology",
    "The mechanism of volcanic eruptions",

    "The characteristics of Renaissance art and architecture",
    "The history of Jazz music and its subgenres",
    "The development of the printing press by Johannes Gutenberg",
    "The philosophy of Stoicism: origins and core beliefs",
    "The literary works of William Shakespeare and their influence",
    "The history of cinema: from silent films to the digital age",
    "The cultural significance of the Olympic Games throughout history",
    "Impressionism in visual arts: style and notable artists",
    "The architecture of Gothic cathedrals",

    "The Great Depression: economic causes and global impact",
    "The structure and function of the United Nations",
    "The history of currency: from barter to digital money",
    "The Industrial Revolution and urbanization",
    "The psychology of cognitive biases",
    "The legal principles of the Magna Carta",
    "The sociology of urbanization in the 21st century",
    "The history of epidemiology and disease control",

    "The life and scientific contributions of Marie Curie",
    "The military strategies of Alexander the Great",
    "The inventions and notebooks of Leonardo da Vinci",
    "The political philosophy of Nelson Mandela",
    "The voyages of Christopher Columbus and the Age of Discovery"
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
        for topic in topics_wiki:
            prompt = PROMPT_WIKI.format(topic=topic)
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=generate_config
                )

                f.write(f"TYPE: ESSAY | TOPIC: {topic} ---\n")
                f.write(response.text + "\n\n")
                print(f"    Esej: {topic[:20]}...", end="\r")
                time.sleep(30)
            except Exception as e:
                print(f"\n    Błąd: {e}")
                time.sleep(10)

    print(f"\nUKOŃCZONO: {filename}")

print("\n--- GOTOWE ---")
