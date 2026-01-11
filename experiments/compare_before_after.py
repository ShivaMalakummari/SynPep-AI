import torch
from binding_scorer import binding_score
from pipeline_generate_and_rank import generate_peptide
from models.peptide_transformer import PeptideTransformer

def avg_score(model, n=30):
    scores = []
    for _ in range(n):
        pep = generate_peptide(model)
        scores.append(binding_score(pep))
    return sum(scores)/len(scores)

base = PeptideTransformer()
base.load_state_dict(torch.load("peptide_model.pt"))

rl = PeptideTransformer()
rl.load_state_dict(torch.load("peptide_model_rl.pt"))

print("Before RL:", avg_score(base))
print("After RL :", avg_score(rl))
