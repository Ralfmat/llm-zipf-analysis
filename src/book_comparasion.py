import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
import os
import random
import powerlaw

FILE_AI = "../data/AI/novel_llama-3.3-70b_temp_1.0.txt"
FILE_HUMAN = "../data/Human/Dune.txt"
MODEL_NAME = "llama-3.3-70b"
NUM_SAMPLES = 10
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


print(f"1. Analiza AI ({FILE_AI})...")
if not os.path.exists(FILE_AI):
    print(f"Brak pliku AI: {FILE_AI}")
    exit()

with open(FILE_AI, 'r', encoding='utf-8') as f:
    ai_text = f.read()

ai_tokens = get_tokens(ai_text)
ai_count = len(ai_tokens)

ai_ranks, ai_probs, ai_unique, ai_alpha = calculate_zipf_and_alpha(ai_tokens)
print(f"   -> AI Liczba słów (N): {ai_count}")
print(f"   -> AI Unikalnych słów: {ai_unique}")
print(f"   -> AI Alpha: {ai_alpha:.4f}")

print(f"2. Analiza {NUM_SAMPLES} próbek z Dune (Długość N={ai_count})...")
if not os.path.exists(FILE_HUMAN):
    print(f"Brak pliku Ludzkiego: {FILE_HUMAN}")
    exit()

with open(FILE_HUMAN, 'r', encoding='utf-8', errors="ignore") as f:
    human_text_full = f.read()

human_tokens_all = get_tokens(human_text_full)
total_human_len = len(human_tokens_all)
print(f"   -> Załadowano Dune: {total_human_len} słów.")

if total_human_len < ai_count:
    print("BŁĄD: Tekst ludzki jest krótszy niż tekst AI! Nie można zrobić samplingu.")
    exit()

human_samples = []
human_alphas = []
human_unique_counts = []
max_rank_global = 0

for i in range(NUM_SAMPLES):
    print(f"   -> Próbka {i + 1}/{NUM_SAMPLES}...", end='\r')

    max_start = total_human_len - ai_count
    start_idx = random.randint(0, max_start)

    sample_tokens = human_tokens_all[start_idx: start_idx + ai_count]

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

start_prob = ai_probs[0]
ideal_ranks = np.arange(1, max_rank_global + 1000)
ideal_probs = start_prob / ideal_ranks

print("\n3. Generowanie wykresów...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.suptitle(f"Analiza: AI vs Dune (Długość próbki N={ai_count})", fontsize=16)

ax1 = axes[0, 0]
for i, (ranks, probs, _) in enumerate(human_samples):
    label_text = f"Próbki Dune (N={ai_count})" if i == 0 else "_nolegend_"
    ax1.loglog(ranks, probs, color='grey', alpha=0.4, linewidth=1, label=label_text)

ax1.loglog(ideal_ranks, ideal_probs, 'r--', linewidth=2, label='Idealny Zipf')
ax1.set_title(f"1. Rozkład tekstu - Frank Herbert (Dune)", weight='bold')
ax1.set_ylabel("Prawdopodobieństwo (Log)")
ax1.grid(True, alpha=0.2)
ax1.legend()

ax2 = axes[0, 1]
ax2.loglog(ai_ranks, ai_probs, color='blue', linewidth=2,
           label=f'AI (Alpha={ai_alpha:.2f}, N={ai_count})')
ax2.loglog(ideal_ranks, ideal_probs, 'r--', linewidth=2, label='Idealny Zipf')
ax2.set_title(f"2. Rozkład tekstu - {MODEL_NAME}", weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.2)

ax3 = axes[1, 0]
ax3.loglog(human_samples[0][0], human_samples[0][1], color='grey', alpha=0.3, linewidth=1,
           label=f'Dune (Śr. unikalnych: {avg_human_unique}, N={ai_count})')
for ranks, probs, _ in human_samples[1:]:
    ax3.loglog(ranks, probs, color='grey', alpha=0.3, linewidth=1)

ax3.loglog(ai_ranks, ai_probs, color='blue', linewidth=3,
           label=f'AI (Unikalnych: {ai_unique}, N={ai_count})')

ax3.loglog(ideal_ranks, ideal_probs, 'r--', linewidth=2, label='Idealny Zipf')

ax3.set_title("3. AI vs Herbert", weight='bold')
ax3.set_xlabel("Ranga słowa")
ax3.set_ylabel("Prawdopodobieństwo")
ax3.grid(True, alpha=0.2)
ax3.legend(fontsize=11, loc='lower left', frameon=True, shadow=True)

ax4 = axes[1, 1]
labels = ['Idealny Zipf', 'Dune (Średnia)', 'AI']
values = [2.0, avg_human_alpha, ai_alpha]
colors = ['green', 'grey', 'blue']

bars = ax4.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=12, weight='bold')

ax4.set_title("4. Parametry ALPHA", weight='bold')
ax4.set_ylabel("Wartość Alpha")
ax4.set_ylim(bottom=1.0)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
