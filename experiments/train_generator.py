import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from models.peptide_transformer import PeptideTransformer
from data_processing.load_fasta import load_fasta
from data_processing.dataset import PeptideDataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sequences = load_fasta("data/egfr/sequences/egfr_positive.fasta")
    dataset = PeptideDataset(sequences)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = PeptideTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(5):
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f}")

    # save trained model
    torch.save(model.state_dict(), "peptide_model.pt")


if __name__ == "__main__":
    main()

