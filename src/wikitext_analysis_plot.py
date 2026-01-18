import powerlaw
from datasets import load_from_disk
from collections import Counter
import re
import matplotlib.pyplot as plt
import numpy as np

# 1. ŁADOWANIE DANYCH
print("1. Inicjalizacja: Wczytywanie WikiText-103...")
try:
    dataset = load_from_disk("../wikitext_103_raw")
except:
    print("Błąd: Nie znaleziono danych w folderze ./wikitext_103_raw")
    exit()

def get_word_counts(dataset_split):
    counter = Counter()
    tokenizer = re.compile(r'\b\w+\b')
    for text in dataset_split['text']:
        if text.strip():
            words = tokenizer.findall(text.lower())
            counter.update(words)
    return counter

# 2. PRZETWARZANIE DANYCH
print("2. Przetwarzanie: Obliczanie częstotliwości i rang...")
word_counts = get_word_counts(dataset['train'])
frequencies = np.array(list(word_counts.values()))
sorted_frequencies = np.sort(frequencies)[::-1]
ranks = np.arange(1, len(sorted_frequencies) + 1)

# 3. KALIBRACJA MODELU
print("3. Analiza: Estymacja parametrów Alpha (Powerlaw)...")
fit = powerlaw.Fit(frequencies, discrete=True, xmin=10)
alpha = fit.power_law.alpha

# 4. GENEROWANIE LINII ODNIESIENIA
# Theoretical Zipf: f(r) = C / r^s (gdzie s=1 dla idealnego Zipfa)
C = sorted_frequencies[0]
ideal_zipf = C / ranks

# 5. WIZUALIZACJA
plt.figure(figsize=(12, 8))

plt.loglog(ranks, sorted_frequencies,
           color='#003366',
           marker='.',
           linestyle='None',
           markersize=1.5,
           alpha=0.6,
           label=f'WikiText-103 (Empirical Data, $\\alpha$={alpha:.2f})')

plt.loglog(ranks, ideal_zipf,
           color='#CC0000',
           linestyle='--',
           linewidth=2,
           label='Theoretical Zipf Law ($s = -1.0$)')

plt.title("Zipf's Law Distribution Analysis: Human Baseline (WikiText-103)", fontsize=16, weight='bold')
plt.xlabel("Word Rank ($r$) [Log Scale]", fontsize=12)
plt.ylabel("Term Frequency ($f$) [Log Scale]", fontsize=12)

plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
plt.grid(True, which="both", ls="-", alpha=0.15)


print("Generowanie wykresu do dokumentacji...")
plt.tight_layout()
plt.savefig("wikitext_baseline_official.png", dpi=300)
plt.show()