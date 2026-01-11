from binding_scorer import binding_score

peptides = [
    "ALLALTYLALGALFGTYLVDGLALSLLQALLALLALALGALALALLLLALGALFGTLGQN",
    "AVHMLTLDGQNTPRYQGVRYAVDYLSLHLHNQPRYLRSLALALDGALFVDVDRDDPLAVD",
    "ANEDGKQTPALAILFGVDPT"
]

results = []

for pep in peptides:
    score = binding_score(pep)
    results.append((pep, score))

results.sort(key=lambda x: x[1], reverse=True)

for pep, score in results:
    print(f"Score: {score} | Peptide: {pep}")
