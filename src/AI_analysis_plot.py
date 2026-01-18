import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
import os

# --- KONFIGURACJA ---
AI_FILES = {
    "AI Temp 0.2 (Konserwatywny)": "ai_text_temp_0.2.txt",
    "AI Temp 0.7 (Standard)": "ai_text_temp_0.7.txt",
    "AI Temp 1.5 (Kreatywny)": "ai_text_temp_1.5.txt"
}

COLORS = {
    "AI Temp 0.2 (Konserwatywny)": "blue",
    "AI Temp 0.7 (Standard)": "orange",
    "AI Temp 1.5 (Kreatywny)": "red"
}


def get_zipf_data(filename):
    """Wczytuje plik i zwraca dane do wykresu"""
    if not os.path.exists(filename):
        print(f"[!] Błąd: Nie znaleziono pliku {filename}")
        return None, None, 0

    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Prosty tokenizator
    tokenizer = re.compile(r'\b\w+\b')
    words = tokenizer.findall(text.lower())

    counter = Counter(words)

    # Sortowanie malejące
    counts = np.array(list(counter.values()))
    counts = np.sort(counts)[::-1]

    # Normalizacja do prawdopodobieństwa
    total_words = np.sum(counts)
    probs = counts / total_words
    ranks = np.arange(1, len(probs) + 1)

    return ranks, probs, total_words


# --- RYSOWANIE ---
plt.figure(figsize=(12, 8))

# Zmienna do przechowania punktu startowego dla linii idealnej
start_prob_ref = 0

print("Analiza plików AI...")

for label, filename in AI_FILES.items():
    ranks, probs, total = get_zipf_data(filename)

    if ranks is not None:
        # Rysujemy linię dla danej temperatury
        plt.loglog(ranks, probs,
                   color=COLORS[label],
                   linewidth=2,
                   linestyle='-',
                   marker='.', markersize=4,
                   label=f'{label} ({total} słów)')

        # Pobieramy najwyższe prawdopodobieństwo (dla słowa "the") z pierwszego pliku
        # żeby dopasować idealną linię Zipfa
        if start_prob_ref == 0 and len(probs) > 0:
            start_prob_ref = probs[0]

# --- RYSOWANIE IDEALNEJ LINII ODNIESIENIA ---
# Nawet bez ludzi warto mieć linię matematyczną, żeby widzieć czy AI "trzyma poziom"
if start_prob_ref > 0:
    # Generujemy linię idealną na długość najdłuższego tekstu (np. do rangi 10000)
    ideal_ranks = np.arange(1, 10000)
    ideal_probs = start_prob_ref / ideal_ranks
    plt.loglog(ideal_ranks, ideal_probs, 'k--', linewidth=1, alpha=0.5, label='Idealne Prawo Zipfa (Teoria)')

# --- KOSMETYKA ---
plt.title("Wpływ Temperatury na Słownictwo AI (Zipf Analysis)", fontsize=16)
plt.xlabel("Ranga słowa (Log Scale)", fontsize=12)
plt.ylabel("Częstotliwość (Log Scale)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="-", alpha=0.2)

# Adnotacje
plt.text(1.5, 0.08, "NAJCZĘSTSZE SŁOWA\n(Tu linie się pokryją)", fontsize=9)
plt.text(500, 0.0001, "OGON\n(Tu szukaj różnic w długości)", fontsize=9, color='darkred')

plt.tight_layout()
plt.show()