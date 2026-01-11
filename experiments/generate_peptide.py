import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.peptide_transformer import PeptideTransformer
from data_processing.tokenizer import decode, PAD_TOKEN

MAX_LEN = 60


def generate_peptide(model, start_token=1, temperature=1.0):
    model.eval()
    seq = [start_token]

    for _ in range(MAX_LEN - 1):
        x = torch.tensor([seq])

        with torch.no_grad():
            logits = model(x)[0, -1]

        # temperature scaling
        logits = logits / temperature
        probs = torch.softmax(logits, dim=0)

        # sample instead of argmax
        next_token = torch.multinomial(probs, 1).item()

        if next_token == PAD_TOKEN:
            break

        seq.append(next_token)

    return decode(seq)


if __name__ == "__main__":
    model = PeptideTransformer()
    model.load_state_dict(torch.load("peptide_model.pt", map_location="cpu"))
    print("🧬 Generated peptide:")
    print("Temp 0.7:", generate_peptide(model, temperature=0.7))
    print("Temp 1.0:", generate_peptide(model, temperature=1.0))
    print("Temp 1.3:", generate_peptide(model, temperature=1.3))


