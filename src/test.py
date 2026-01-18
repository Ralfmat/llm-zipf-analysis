import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from collections import Counter
import re
import os
import random
import powerlaw

# KONFIGURACJA
AI_FILE = "../data/AI/gemini_temp_0.2.txt"
MODEL_NAME = "gemini-2.5-flash"
NUM_SAMPLES = 1
XMIN_SETTING = 1


def get_tokens(text):
    tokenizer = re.compile(r'\b\w+\b')
    return tokenizer.findall(text.lower())


def calculate_zipf_and_alpha(tokens):
    counter = Counter(tokens)
    counts = np.array(list(counter.values()))
    counts = np.sort(counts)[::-1]
    total = np.sum(counts)
    probs = counts / total
    ranks = np.arange(1, len(probs) + 1)

    try:
        fit = powerlaw.Fit(counts, discrete=True, xmin=XMIN_SETTING, verbose=False)
        alpha = fit.power_law.alpha
    except:
        alpha = 0

    return ranks, probs, len(probs), alpha


# PRZYGOTOWANIE DANYCH AI
print(f"1. Analiza AI ({AI_FILE})...")
if not os.path.exists(AI_FILE):
    print("Brak pliku!")
    exit()

with open(AI_FILE, 'r', encoding='utf-8') as f:
    ai_text = f.read()
ai_tokens = get_tokens(ai_text)
ai_count = len(ai_tokens)

ai_ranks, ai_probs, ai_unique, ai_alpha = calculate_zipf_and_alpha(ai_tokens)
print(f"   -> AI Unikalnych słów: {ai_unique}")
print(f"   -> AI Alpha: {ai_alpha:.4f}")

# PRZYGOTOWANIE DANYCH LUDZKICH
print(f"2. Analiza {NUM_SAMPLES} próbek WikiText...")
try:
    dataset = load_from_disk("../wikitext_103_raw")
    full_human_text = dataset['train']['text']
except:
    print("Błąd WikiText.")
    exit()

human_samples = []
human_alphas = []
human_unique_counts = []
max_rank_global = 0

for i in range(NUM_SAMPLES):
    print(f"   -> Próbka {i + 1}...", end='\r')
    random_lines = random.sample(full_human_text, 5000)
    sample_tokens = []
    for line in random_lines:
        if line.strip():
            sample_tokens.extend(get_tokens(line))
            if len(sample_tokens) >= ai_count:
                break
    sample_tokens = sample_tokens[:ai_count]

    h_ranks, h_probs, h_unique, h_alpha = calculate_zipf_and_alpha(sample_tokens)

    human_samples.append((h_ranks, h_probs, h_unique))
    human_alphas.append(h_alpha)
    human_unique_counts.append(h_unique)

    if len(h_ranks) > max_rank_global:
        max_rank_global = len(h_ranks)

avg_human_alpha = np.mean(human_alphas)
avg_human_unique = int(np.mean(human_unique_counts))
print(f"\n   -> Średnia Ludzka Alpha: {avg_human_alpha:.4f}")
print(f"   -> Średnia liczba unikalnych słów (Człowiek): {avg_human_unique}")

# Idealny Zipf
start_prob = ai_probs[0]
ideal_ranks = np.arange(1, max_rank_global + 2000)
ideal_probs = start_prob / ideal_ranks

# RYSOWANIE DASHBOARDA
print("\n3. Generowanie wykresów...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# WYKRES 1: LUDZIE
ax1 = axes[0, 0]
for ranks, probs, _ in human_samples:
    ax1.loglog(ranks, probs, color='grey', alpha=0.4, linewidth=1)
ax1.loglog(ideal_ranks, ideal_probs, 'r--', linewidth=2, label='Idealny Zipf')
ax1.set_title(f"1. Rozkład ludzkiego tekstu", weight='bold')
ax1.set_ylabel("Prawdopodobieństwo (Log)")
ax1.grid(True, alpha=0.2)
ax1.legend()

# WYKRES 2: AI
ax2 = axes[0, 1]
ax2.loglog(ai_ranks, ai_probs, color='blue', linewidth=2, label=f'AI (Alpha={ai_alpha:.2f})')
ax2.loglog(ideal_ranks, ideal_probs, 'r--', linewidth=2, label='Idealny Zipf')
ax2.set_title(f"2. Rozkład tekstu SI {MODEL_NAME}", weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.2)

# WYKRES 3: KONFRONTACJA
ax3 = axes[1, 0]

ax3.loglog(human_samples[0][0], human_samples[0][1], color='grey', alpha=0.3, linewidth=1,
           label=f'Człowiek (Śr. unikalnych: {avg_human_unique})')
for ranks, probs, _ in human_samples[1:]:
    ax3.loglog(ranks, probs, color='grey', alpha=0.3, linewidth=1)

ax3.loglog(ai_ranks, ai_probs, color='blue', linewidth=3,
           label=f'AI (Unikalnych: {ai_unique})')

ax3.loglog(ideal_ranks, ideal_probs, 'r--', linewidth=2, label='Idealny Zipf')

ax3.set_title("3. Nałożenie rozkładu tekstu ludzkiego i SI", weight='bold')
ax3.set_xlabel("Ranga słowa")
ax3.set_ylabel("Prawdopodobieństwo")
ax3.grid(True, alpha=0.2)

ax3.legend(fontsize=11, loc='lower left', frameon=True, shadow=True)

# WYKRES 4: Alphy
ax4 = axes[1, 1]
labels = ['Idealny Zipf', 'Człowiek', 'AI']
values = [2.0, avg_human_alpha, ai_alpha]
colors = ['green', 'grey', 'blue']
bars = ax4.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
ax4.axhline(y=2.0, color='green', linestyle='--', linewidth=1)

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=12, weight='bold')

# ax4.set_ylim(1.5, 2.3)  # Skalujemy, żeby było widać różnice przy xmin=1
ax4.set_title("4. Parametry ALPHA dla tekstu ludzkiego i SI", weight='bold')
ax4.set_ylabel("Wartość Alpha")

plt.tight_layout()
plt.show()
