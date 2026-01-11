import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.peptide_transformer import PeptideTransformer
from data_processing.tokenizer import decode, PAD_TOKEN
from binding_scorer import binding_score

MAX_LEN = 60
NUM_SAMPLES = 50


def generate_peptide(model, temperature=1.0):
    seq = [1]  # start token
    model.eval()

    for _ in range(MAX_LEN - 1):
        x = torch.tensor([seq])
        with torch.no_grad():
            logits = model(x)[0, -1]

        probs = torch.softmax(logits / temperature, dim=0)
        next_token = torch.multinomial(probs, 1).item()

        if next_token == PAD_TOKEN:
            break

        seq.append(next_token)

    return decode(seq)


if __name__ == "__main__":
    model = PeptideTransformer()
    model.load_state_dict(torch.load("peptide_model.pt", map_location="cpu"))

    results = []

    for _ in range(NUM_SAMPLES):
        pep = generate_peptide(model, temperature=1.0)
        score = binding_score(pep)
        results.append((pep, score))

    results.sort(key=lambda x: x[1], reverse=True)

    for pep, score in results[:10]:
        print(f"Score: {score} | {pep}")
