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

print('ATTENTION')

# Minimal cross attention network
class AttentionModel(nn.Module):
  def __init__(self, plm, PLM):
    super(AttentionModel, self).__init__()
    self.plm = plm
    self.PLM = PLM
    self.attn = nn.MultiheadAttention(320, 1, batch_first=True)
    self.prediction_head = torch.nn.Sequential(
        torch.nn.Linear(320, 2),
        # torch.nn.ReLU(),
        # torch.nn.Linear(128, 2),
        torch.nn.Softmax(dim=1)
    )
    self.post_attn_norm = nn.LayerNorm(320)
    # Freeze the anchor and augment models
    for param in self.plm.parameters(): param.requires_grad = False
    for param in self.PLM.parameters(): param.requires_grad = False

  def forward(self, pep_tokens, pro_tokens, attn_mask=None):

    # Obtain protein sequence embedding L x D. len(pro_tokens.input_ids) is len(protein_sequence)+2
    with torch.no_grad():
        results = self.PLM.forward(pro_tokens['input_ids'], pro_tokens['attention_mask'], output_hidden_states=True)
    t1 = results.hidden_states[-1]

    # Obtain peptide sequence embedding L x D. len(pep_tokens.input_ids) is len(peptide_sequence)+2
    with torch.no_grad():
        results = self.plm.forward(pep_tokens['input_ids'], pep_tokens['attention_mask'], output_hidden_states=True)
    t2 = results.hidden_states[-1]

    # Create attn mask
    attn_mask = torch.matmul(
        pro_tokens["attention_mask"].unsqueeze(2).float(),
        pep_tokens['attention_mask'].unsqueeze(1).float(),
    ).repeat(1, 1, 1)

    # Perform attention
    output, attn_weights = self.attn(
        query=t1, key=t2, value=t2, attn_mask=attn_mask, average_attn_weights=False
    )
    output = self.post_attn_norm(output) + t1

    # Take the mean. Exclude padding tokens and bos/eos
    mask_sum = pro_tokens['attention_mask'].sum(dim=1, keepdim=True).clamp(min=1e-6)  # Avoid division by zero
    output = (output * pro_tokens['attention_mask'].unsqueeze(2)).sum(dim=1) / mask_sum

    return self.prediction_head(output), output

for n in range(3):
    # Define model and other components
    model = AttentionModel(plm, PLM)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_test_acc = 0
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_dataloader:
            # Forward pass
            _, _, labels, pep_tokens, pro_tokens = batch
            labels = labels.to(device)
            outputs, _ = model(pep_tokens, pro_tokens)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Compute classification metrics
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
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

                # Forward pass
                _, _, labels, pep_tokens, pro_tokens = batch
                labels = labels.to(device)
                outputs, _ = model(pep_tokens, pro_tokens)
                loss = criterion(outputs, labels)

                # Compute classification metrics
                test_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            test_loss /= total
            test_accuracy = correct / total

        # Save if best
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            torch.save(model.state_dict(), "attention_model.pth")
        if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(best_test_acc)

#@title Visualization for attention
model = AttentionModel(plm, PLM)
model = model.to(device)
model.load_state_dict(torch.load("./attention_model.pth", weights_only=True))
model.eval()

embs_attention = []
y_true = []
with torch.no_grad():
    for batch in test_dataloader:
        _, _, labels, pep_tokens, pro_tokens = batch
        labels = labels.to(device)
        pep_out, _ = model(pep_tokens, pro_tokens)
        for example, y in zip(pep_out, labels):
            embs_attention.append(example.detach().cpu().numpy())
            y_true.append(y.item())
embs_attention = np.array(embs_attention)

# Perform PCA
embs_attention = np.array(embs_attention)
tsne = TSNE(n_components=2, perplexity=10, random_state=1)
embs_attention = tsne.fit_transform(embs_attention)

# Colormap
cm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["royalblue","crimson"])

# Plot
plt.figure(figsize=(6, 6))
scatter = plt.scatter(embs_attention[:, 0], embs_attention[:, 1], c=y_true, cmap=cm, s=10, alpha=0.9)
plt.xlabel("t-SNE-1", fontsize=18)
plt.ylabel("t-SNE-2", fontsize=18)
plt.scatter([], [], c='crimson', edgecolor='k', alpha=0.7, label='Binding pair')
plt.scatter([], [], c='royalblue', edgecolor='k', alpha=0.7, label='Non-binding pair')
plt.scatter([], [], c='mediumseagreen', edgecolor='k', alpha=0.7, label='Non-binding pair')
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('./attention.png', dpi=600, bbox_inches='tight')
plt.close()
