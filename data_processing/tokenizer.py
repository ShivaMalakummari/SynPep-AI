AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

aa_to_idx = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
idx_to_aa = {i + 1: aa for i, aa in enumerate(AMINO_ACIDS)}

PAD_TOKEN = 0

def encode(sequence, max_len=60):
    encoded = [aa_to_idx[aa] for aa in sequence if aa in aa_to_idx]
    return encoded[:max_len] + [PAD_TOKEN] * (max_len - len(encoded))

def decode(encoded_seq):
    return "".join(idx_to_aa.get(i, "") for i in encoded_seq if i != PAD_TOKEN)
