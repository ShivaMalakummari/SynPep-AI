import sys, os, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.peptide_transformer import PeptideTransformer
from data_processing.tokenizer import decode, PAD_TOKEN
from binding_scorer import binding_score

MAX_LEN = 60
LR = 1e-4
STEPS = 50


def generate_with_logprobs(model):
    seq = [1]
    log_probs = []

    for _ in range(MAX_LEN - 1):
        x = torch.tensor([seq])
        logits = model(x)[0, -1]
        probs = torch.softmax(logits, dim=0)

        dist = torch.distributions.Categorical(probs)
        token = dist.sample()

        log_probs.append(dist.log_prob(token))

        if token.item() == PAD_TOKEN:
            break

        seq.append(token.item())

    return decode(seq), torch.stack(log_probs).sum()


if __name__ == "__main__":
    model = PeptideTransformer()
    model.load_state_dict(torch.load("peptide_model.pt"))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(STEPS):
        peptide, logp = generate_with_logprobs(model)
        reward = binding_score(peptide)

        loss = -logp * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step} | Reward {reward} | {peptide}")

    torch.save(model.state_dict(), "peptide_model_rl.pt")
