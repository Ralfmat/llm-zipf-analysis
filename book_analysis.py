import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
import os
import powerlaw


INPUT_FILE = "Dune.txt"
XMIN_SETTING = 1


def get_words(text):
    """Czyści tekst i dzieli go na słowa."""
    tokenizer = re.compile(r'\b\w+\b')
    return tokenizer.findall(text.lower())


def calculate_stats(tokens):
    """Oblicza rangi, prawdopodobieństwa i parametr Alpha."""
    counter = Counter(tokens)
    counts = np.array(list(counter.values()))

    counts = np.sort(counts)[::-1]

    total_words = np.sum(counts)
    probs = counts / total_words
    ranks = np.arange(1, len(probs) + 1)

    try:
        fit = powerlaw.Fit(counts, discrete=True, xmin=XMIN_SETTING, verbose=False)
        alpha = fit.power_law.alpha
    except Exception as e:
        print(f"Nie udało się obliczyć Alpha: {e}")
        alpha = 0

    return ranks, probs, len(probs), alpha, counter


print(f"1. Wczytywanie pliku: {INPUT_FILE}...")
if not os.path.exists(INPUT_FILE):
    print(f"BŁĄD: Nie znaleziono pliku '{INPUT_FILE}'. Sprawdź nazwę.")
    exit()

with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

print("2. Przetwarzanie tekstu...")
tokens = get_words(text)
total_tokens = len(tokens)

if total_tokens == 0:
    print("Plik jest pusty lub nie zawiera słów.")
    exit()

ranks, probs, unique_count, alpha, counter = calculate_stats(tokens)

print(f"   -> Liczba wszystkich słów: {total_tokens}")
print(f"   -> Liczba unikalnych słów: {unique_count}")
print(f"   -> Obliczony parametr Alpha: {alpha:.4f}")


ideal_ranks = ranks
ideal_probs = probs[0] / ideal_ranks  # Prawo Zipfa: P(r) ~ 1/r

# --- RYSOWANIE WYKRESÓW ---
print("3. Generowanie wykresu...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# WYKRES 1: Rozkład Zipfa (Log-Log)
ax1.loglog(ranks, probs, color='blue', linewidth=2, label=f'Książka (α={alpha:.2f})')
ax1.loglog(ideal_ranks, ideal_probs, 'r--', linewidth=2,
           label='Idealny Zipf (α=~2.0)')

ax1.set_title(f"Rozkład Zipfa: {INPUT_FILE}", fontsize=14, weight='bold')
ax1.set_xlabel("Ranga słowa (log)", fontsize=12)
ax1.set_ylabel("Częstość występowania (log)", fontsize=12)
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.legend(fontsize=11)

top_n = 15
most_common = counter.most_common(top_n)
words = [w[0] for w in most_common]
counts_top = [w[1] for w in most_common]

ax2.bar(words, counts_top, color='skyblue', edgecolor='black')
ax2.set_title(f"Top {top_n} najczęstszych słów", fontsize=14, weight='bold')
ax2.set_ylabel("Liczba wystąpień", fontsize=12)
ax2.tick_params(axis='x', rotation=45)

# Dodanie statystyk tekstowych na wykresie
info_text = (
    f"Plik: {INPUT_FILE}\n"
    f"Wszystkich słów: {total_tokens}\n"
    f"Unikalnych: {unique_count}\n"
    f"Alpha: {alpha:.4f}"
)
# Ramka z informacjami w prawym górnym rogu drugiego wykresu
ax2.text(0.95, 0.95, info_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()