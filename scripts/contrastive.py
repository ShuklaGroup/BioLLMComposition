import math
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.nn import CosineSimilarity
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 1e-3
EPOCHS = 100
VERBOSE = False

# Load MHC data set 
MHCtrain = pd.read_csv('./combo_1and2_train.tsv', sep='\t')
MHCval = pd.read_csv('./combo_1and2_valid.tsv', sep='\t')

# Extract sequences from data frames
train_seqs = MHCtrain['target_chainseq'].tolist()
train_labs = MHCtrain['binder'].tolist()
val_seqs = MHCval['target_chainseq'].tolist()
val_labs = MHCval['binder'].tolist()

# Preprocess sequences as the protein and peptide are separated by '/'
mhc_train_pep = []
mhc_train_rec = []
mhc_train_lab = []
for i, s in enumerate(train_seqs):
    try:
        rec, pep = s.split('/')
        mhc_train_pep.append(pep)
        mhc_train_rec.append(rec)
        mhc_train_lab.append(train_labs[i])
    except:
        pass

mhc_val_pep = []
mhc_val_rec = []
mhc_val_lab = []
for i, s in enumerate(val_seqs):
    try:
        rec, pep = s.split('/')
        mhc_val_pep.append(pep)
        mhc_val_rec.append(rec)
        mhc_val_lab.append(val_labs[i])
    except:
        pass

# Load pLM and PLM models
esm_layers = 6
esm_params = 8
plm = AutoModelForMaskedLM.from_pretrained(f'facebook/esm2_t{esm_layers}_{esm_params}M_UR50D').to(device).eval()
PLM = AutoModelForMaskedLM.from_pretrained(f'facebook/esm2_t{esm_layers}_{esm_params}M_UR50D').to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(f'facebook/esm2_t{esm_layers}_{esm_params}M_UR50D')

# Get the mean embedding from esm style model
def get_mean_rep(model_name, sequence):
    token_ids = tokenizer(sequence, return_tensors='pt').to(device)
    with torch.no_grad():
        results = model_name.forward(token_ids.input_ids, output_hidden_states=True)
    representations = results.hidden_states[-1][0]
    mean_embedding = representations[1:len(sequence)+1].mean(dim=0)
    return mean_embedding.cpu().numpy()
    
# Custom data set class for peptide-protein pairs
class mhcdataset(Dataset):
    def __init__(self, peptides, proteins, labels, p_tokens, P_tokens):
        self.peptides = peptides # Peptide sequenes
        self.proteins = proteins # Protein sequences
        self.labels = labels # Binary labels
        self.p_tokens = p_tokens # ESM tokens for the sequences
        self.P_tokens = P_tokens

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        p = self.peptides[i]
        P = self.proteins[i]
        y = self.labels[i]
        p_tokens = {key: value[i] for key, value in self.p_tokens.items()}
        P_tokens = {key: value[i] for key, value in self.P_tokens.items()}
        return p, P, y, p_tokens, P_tokens

# Extract training data set embeddings
peptide_embs_train, protein_embs_train = [], []
for pep, pro in tqdm(zip(mhc_train_pep, mhc_train_rec)):
    peptide_embs_train.append(get_mean_rep(plm, pep))
    protein_embs_train.append(get_mean_rep(PLM, pro))
peptide_embs_train, protein_embs_train = np.array(peptide_embs_train), np.array(protein_embs_train)

# Extract test data set embeddings
peptide_embs_val, protein_embs_val = [], []
for pep, pro in tqdm(zip(mhc_val_pep, mhc_val_rec)):
    peptide_embs_val.append(get_mean_rep(plm, pep))
    protein_embs_val.append(get_mean_rep(PLM, pro))
peptide_embs_val, protein_embs_val = np.array(peptide_embs_val), np.array(protein_embs_val)

# Tokenize
pep_tokens_train = tokenizer(mhc_train_pep, return_tensors='pt', padding='max_length', max_length=9, truncation=True).to(device)
pro_tokens_train = tokenizer(mhc_train_rec, return_tensors='pt', padding='max_length', max_length=181, truncation=True).to(device)
pep_tokens_val = tokenizer(mhc_val_pep, return_tensors='pt', padding='max_length', max_length=9, truncation=True).to(device)
pro_tokens_val = tokenizer(mhc_val_rec, return_tensors='pt', padding='max_length', max_length=181, truncation=True).to(device)

# Make data sets
train_data_set = mhcdataset(peptide_embs_train, protein_embs_train, np.array(mhc_train_lab), pep_tokens_train, pro_tokens_train)
train_dataloader = DataLoader(train_data_set, batch_size=128, shuffle=True)
test_data_set = mhcdataset(peptide_embs_val, protein_embs_val, np.array(mhc_val_lab), pep_tokens_val, pro_tokens_val)
test_dataloader = DataLoader(test_data_set, batch_size=128, shuffle=True)

print('CONTRASTIVE LEARNING')

# Simple contrastive learning model
class CLModel(nn.Module):
  def __init__(self):
    super(CLModel, self).__init__()

    # Projection network for peptide
    self.peptide_proj = torch.nn.Sequential(
      torch.nn.Linear(320, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 128),
    )

    # Projection network for protein
    self.protein_proj = torch.nn.Sequential(
      torch.nn.Linear(320, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 128),
    )

  def forward(self, pep, pro):
    # Project and normalize
    pep = self.peptide_proj(pep)
    pro = self.protein_proj(pro)
    pep = F.normalize(pep, p=2, dim=1)
    pro = F.normalize(pro, p=2, dim=1)
    return pep, pro

# Train the model 3 independent times. Report the highest performance on the test set each time.
for n in range(3):

    # Define model and other components
    model = CLModel()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CosineEmbeddingLoss()

    best_test_acc = 0
    for epoch in tqdm(range(EPOCHS)):

        # Start training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Load data for batch
        for batch in train_dataloader:
            peptide_embs, protein_embs, labels, _, _ = batch

            # Replace zeros with -1s for cosine loss
            labels[labels == 0] = -1

            # Forward pass
            peptide_embs, protein_embs, labels = peptide_embs.to(device), protein_embs.to(device), labels.to(device)
            pep_out, pro_out = model(peptide_embs, protein_embs)

            # Cosine similarity loss
            loss = criterion(pep_out, pro_out, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Get classification metrics via cosine similarity
            train_loss += loss.item() * labels.size(0)
            similarities = CosineSimilarity(dim=1, eps=1e-6)(pep_out, pro_out)
            preds = (similarities > 0).long() * 2 - 1  # Map to -1 or 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_accuracy = correct / total

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_dataloader:
                peptide_embs, protein_embs, labels, _, _ = batch

                # Replace zeros with -1s for cosine loss
                labels[labels == 0] = -1

                # Forward pass, Cosine similarity loss
                peptide_embs, protein_embs, labels = peptide_embs.to(device), protein_embs.to(device), labels.to(device)
                pep_out, pro_out = model(peptide_embs, protein_embs)
                loss = criterion(pep_out, pro_out, labels)

                # Get classification metrics by cosine sim
                test_loss += loss.item() * labels.size(0)
                similarities = CosineSimilarity(dim=1, eps=1e-6)(pep_out, pro_out)
                preds = (similarities > 0).long() * 2 - 1 # Map to -1 or 1
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            test_loss /= total
            test_accuracy = correct / total

            # Save if model had best accuracy so far
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                torch.save(model.state_dict(), "contrastive_model.pth")
            if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    print(best_test_acc)

# Visualization for contrastive learning

model = CLModel()
model = model.to(device)
model.load_state_dict(torch.load("./contrastive_model.pth", weights_only=True))
model.eval()

embs_contrastive = []
y_true = []
with torch.no_grad():
  for batch in test_dataloader:
    peptide_embs, protein_embs, labels, _, _ = batch
    peptide_embs, protein_embs, labels = peptide_embs.to(device), protein_embs.to(device), labels.to(device)
    pep_out, pro_out = model(peptide_embs, protein_embs)

    for example, y in zip(pep_out, labels):
      embs_contrastive.append(example.detach().cpu().numpy())
      y_true.append(1)

    for example in pro_out:
      embs_contrastive.append(example.detach().cpu().numpy())
      y_true.append(0.5)

# Perform PCA
embs_contrastive = np.array(embs_contrastive)
tsne = TSNE(n_components=2, perplexity=10, random_state=1)
embs_contrastive = tsne.fit_transform(embs_contrastive)

# Colormap
cm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkorange","royalblue"])

# Plot
plt.figure(figsize=(6, 6))
scatter = plt.scatter(embs_contrastive[:, 0], embs_contrastive[:, 1], c=y_true, cmap=cm, s=10, alpha=0.9)
plt.xlabel("t-SNE-1", fontsize=18)
plt.ylabel("t-SNE-2", fontsize=18)
plt.scatter([], [], c='crimson', edgecolor='k', alpha=0.7, label='Binding pair')
plt.scatter([], [], c='royalblue', edgecolor='k', alpha=0.7, label='Non-binding pair')
plt.scatter([], [], c='mediumseagreen', edgecolor='k', alpha=0.7, label='Non-binding pair')
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('./contrastive.png', dpi=600, bbox_inches='tight')
plt.close()
