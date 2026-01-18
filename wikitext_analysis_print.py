from datasets import load_from_disk
from collections import Counter
import re

# 1. ŁADOWANIE DANYCH
print("1. Wczytywanie WikiText-103...")
try:
    dataset = load_from_disk("./wikitext_103_raw")
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

# 2. PRZETWARZANIE
print("2. Liczenie i sortowanie słów...")
word_counts = get_word_counts(dataset['train'])

sorted_vocab = word_counts.most_common()
total_vocab_size = len(sorted_vocab)

print(f"   Całkowita liczba unikalnych słów: {total_vocab_size}")

# 3. WYZNACZANIE PUNKTÓW (INDEKSÓW)
idx_top = 0
idx_third = int(total_vocab_size * 0.3)  # 30% listy (Jedna trzecia)
idx_mid = int(total_vocab_size * 0.5) # 50% listy (Środek)

# Funkcja pomocnicza do ładnego wyświetlania
def print_section(title, start_index, data):
    print(f"\n--- {title} (Rangi {start_index+1} - {start_index+10}) ---")
    print(f"{'RANGA':<10} | {'SŁOWO':<20} | {'CZĘSTOTLIWOŚĆ':<15}")
    print("-" * 50)
    for i in range(10):
        if start_index + i < len(data):
            word, count = data[start_index + i]
            rank = start_index + i + 1
            print(f"{rank:<10} | {word:<20} | {count:<15}")

# 4. WYŚWIETLANIE WYNIKÓW
print_section("Top 10", idx_top, sorted_vocab)

if total_vocab_size > 20:
    print_section("JEDNA TRZECIA (#0% Słownika)", idx_third, sorted_vocab)
    print_section("ŚRODEK (50% Słownika)", idx_mid, sorted_vocab)
