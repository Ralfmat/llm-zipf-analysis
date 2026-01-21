import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_KEY")
client = Groq(api_key=API_KEY)
model_name = "llama-3.3-70b-versatile"
# temperatures = [0.2, 1.0, 1.5]
temperatures = [1.0]

PROMPT_NOVEL = """Write a chapter of a science fiction novel based on the following scene: {topic}.

Rules:
1. Style: High literary fiction. Use complex sentence structures, rich metaphors, and sensory details (sight, sound, smell, heat).
2. Perspective: Third-person limited or First-person. Focus on the character's internal thoughts and physical struggle.
3. Setting: A harsh, arid, desert planet with advanced but decaying technology.
4. Tone: Serious, ominous, political, and mystical.
5. Vocabulary: Avoid simple connecting words. Use specific verbs and descriptive adjectives.
6. Length: Approx 1000 words."""

topics_novel = [
    "A noble family arrives at their new stronghold on a scorching desert planet, greeted by hostile heat and wary locals.",
    "A young heir undergoes a painful test of humanity involving a mysterious box and a nerve-induction poison.",
    "A ducal leader suspects a traitor among his trusted advisors during a tense strategy meeting about resource mining.",
    "An assassination attempt is made on the heir using a small, remote-controlled hunter-seeker drone in his bedroom.",
    "A dinner party where the new rulers try to navigate the complex politics and etiquette of the planet's elite class.",
    "The betrayal: The city's shield wall is sabotaged, and enemy troops descend in the night to slaughter the noble house.",

    "The heir and his mother flee into the deep open desert, struggling to survive a violent sandstorm.",
    "A scene describing the mechanics of a 'stillsuit' or survival suit that recycles the body's moisture.",
    "The fugitives encounter a colossal, subterranean sand-creature that can swallow entire mining factories.",
    "The heir is accepted by a tribe of desert nomads after a ritual knife duel to the death.",
    "A religious ceremony where the mother drinks a poisonous blue liquid to gain ancestral memories.",
    "The heir learns to ride the giant sand-creature using hooks to pry open its armored scales.",

    "The heir adopts a new name and begins a guerrilla war, destroying the enemy's resource harvesting operations.",
    "A vision or hallucination where the heir sees multiple possible futures and realizes the burden of his destiny.",
    "The Emperor arrives on the planet with his elite legions to crush the rebellion once and for all.",
    "The final battle: The nomads use the giant sand-creatures to storm the capital city during a storm.",
    "The duel between the heir and the sadistic nephew of the enemy baron in the throne room.",
    "The heir ascends to the throne, realizing that his holy war will spread across the galaxy against his will."
]

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


print(f"Model: {model_name}")
print(f"Szacowana długość jednego pliku: ~{len(topics) * 800} słów")

for temp in temperatures:
    filename = f"{model_name}_temp_{temp}.txt"
    print(f"\nGenerowanie pliku: {filename} (Temperatura: {temp})...")

    with open(filename, "w", encoding="utf-8") as f:
        for i, topic in enumerate(topics_novel):
            for k in range(10):
                final_prompt = f"write a version{k} of " + PROMPT_NOVEL.format(topic=topic)

                print(f"  [{i + 1}/{len(topics)}] Piszę esej o: {topic[:40]}...", end="\r")

                try:
                    start_time = time.time()

                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": final_prompt}],
                        model=model_name,
                        temperature=temp,
                        max_tokens=16000,
                    )

                    content = response.choices[0].message.content

                    f.write(f"TOPIC: {topic}\n")
                    f.write(content + "\n\n")

                    elapsed = time.time() - start_time

                    time.sleep(3)

                except Exception as e:
                    print(f"\n  [!] Błąd przy temacie '{topic}': {e}")
                    time.sleep(5)

    print(f"\nUKOŃCZONO: {filename}")
