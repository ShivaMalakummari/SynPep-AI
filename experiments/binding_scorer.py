from collections import Counter

HYDROPHOBIC = set("AILMFWV")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
AROMATIC = set("FWY")


def binding_score(peptide: str) -> float:
    if len(peptide) < 10:
        return 0.0

    counts = Counter(peptide)
    length = len(peptide)

    hydrophobic_ratio = sum(counts[a] for a in HYDROPHOBIC) / length
    positive_ratio = sum(counts[a] for a in POSITIVE) / length
    negative_ratio = sum(counts[a] for a in NEGATIVE) / length
    aromatic_ratio = sum(counts[a] for a in AROMATIC) / length
    diversity = len(counts) / length

    # EGFR-inspired heuristic score
    score = (
        2.0 * hydrophobic_ratio +
        1.5 * aromatic_ratio +
        1.2 * diversity -
        0.5 * abs(positive_ratio - negative_ratio)
    )

    return round(score, 4)
