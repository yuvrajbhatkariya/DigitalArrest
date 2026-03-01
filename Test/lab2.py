import string
from collections import Counter

text = input("Enter a text: ")

for char in string.punctuation:
    text = text.replace(char, "")

text = text.lower()

words = text.split()

total_words = len(words)

unique_words = len(set(words))

word_count = Counter(words)

print("Total words:", total_words)
print("Unique words:", unique_words)
print("Most common words:")
print(word_count.most_common(5))  