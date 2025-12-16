#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#@title Load data (~4 min)
#!git clone https://github.com/jjoecclark/BioLLMComposition

import math
import torch
import random
import matplotlib
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR = 5e-4
EPOCHS = 100
VERBOSE = False

# Dataset type to load {full_data}
data_split = 'full_data'
max_smi_len_train = 200 #For splits in order {200, 200, 200}
max_pro_len_train = 1021 #For splits in order {1021, 1021, 1021}
max_smi_len_val = 200 #For splits in order {200, 196, 200}
max_pro_len_val = 1021 #For splits in order {1021, 1021, 1014}


# In[2]:


# Load Protein-Ligand BIOSNAP data
train_data_df = pd.read_csv(f'./train_2k.csv')
val_data_df = pd.read_csv(f'./test_400.csv')

# Extract sequences from data frames
train_seqs = train_data_df['Target Sequence'].tolist()
train_smiles = train_data_df['SMILES'].tolist()
train_labs = train_data_df['Label'].tolist()
val_seqs = val_data_df['Target Sequence'].tolist()
val_smiles = val_data_df['SMILES'].tolist()
val_labs = val_data_df['Label'].tolist()

# Load PLM and CLM models
esm_layers = 6 #6
esm_params = 8 #8
# NOTE: When done, replace back to f'facebook/esm2_...'
PLM = AutoModelForMaskedLM.from_pretrained(f'facebook/esm2_t{esm_layers}_{esm_params}M_UR50D').to(device).eval()
chem_config = AutoConfig.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", output_hidden_states=True, deterministic_eval=True, trust_remote_code=True)
CLM = AutoModelForMaskedLM.from_pretrained('ibm-research/MoLFormer-XL-both-10pct', config=chem_config, trust_remote_code=True).to(device).eval()
PLM_tokenizer = AutoTokenizer.from_pretrained(f'facebook/esm2_t{esm_layers}_{esm_params}M_UR50D')
CLM_tokenizer = tokenizer_smiles = AutoTokenizer.from_pretrained('ibm-research/MoLFormer-XL-both-10pct', trust_remote_code=True)
# NOTE: just do MoLFormer part when done

# Get the mean embedding from esm style model
def get_mean_prot_rep(model_name, sequence):
    token_ids = PLM_tokenizer(sequence, return_tensors='pt').to(device)
    with torch.no_grad():
        results = model_name.forward(token_ids.input_ids, output_hidden_states=True)
    representations = results.hidden_states[-1][0]
    mean_embedding = representations[1:len(sequence)+1].mean(dim=0)
    return mean_embedding.cpu().numpy()

# Get the mean embedding from MolFormer model
def get_mean_chem_rep(model_name, smiles):
    token_ids = CLM_tokenizer(smiles, return_tensors='pt').to(device)
    with torch.no_grad():
        results = model_name.forward(token_ids.input_ids, output_hidden_states=True)
    representations = results.hidden_states[-1][0]
    mean_embedding = representations[1:len(smiles)+1].mean(dim=0)
    return mean_embedding.cpu().numpy()

# Custom data set class for protein-ligand pairs
class biosnapdataset(Dataset):
    def __init__(self, ligands, proteins, labels, L_tokens, P_tokens):
        self.ligands = ligands # ligand SMILES
        self.proteins = proteins # Protein sequences
        self.labels = labels # Binary labels
        self.L_tokens = L_tokens # ESM tokens for the sequences
        self.P_tokens = P_tokens

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        L = self.ligands[i]
        P = self.proteins[i]
        y = self.labels[i]
        L_tokens = {key: value[i] for key, value in self.L_tokens.items()}
        P_tokens = {key: value[i] for key, value in self.P_tokens.items()}
        return L, P, y, L_tokens, P_tokens

# Extract training data set embeddings
ligand_embs_train, protein_embs_train = [], []
for smi, pro in tqdm(zip(train_smiles, train_seqs)):
    ligand_embs_train.append(get_mean_chem_rep(CLM, smi))
    protein_embs_train.append(get_mean_prot_rep(PLM, pro))
ligand_embs_train, protein_embs_train = np.array(ligand_embs_train), np.array(protein_embs_train)

# Extract test data set embeddings
ligand_embs_val, protein_embs_val = [], []
for smi, pro in tqdm(zip(val_smiles, val_seqs)):
    ligand_embs_val.append(get_mean_chem_rep(CLM, smi))
    protein_embs_val.append(get_mean_prot_rep(PLM, pro))
ligand_embs_val, protein_embs_val = np.array(ligand_embs_val), np.array(protein_embs_val)

# Tokenize
lig_tokens_train = CLM_tokenizer(train_smiles, return_tensors='pt', padding='max_length', max_length=(max_smi_len_train+2), truncation=True).to(device)
pro_tokens_train = PLM_tokenizer(train_seqs, return_tensors='pt', padding='max_length', max_length=(max_pro_len_train+2), truncation=True).to(device)
lig_tokens_val = CLM_tokenizer(val_smiles, return_tensors='pt', padding='max_length', max_length=(max_smi_len_val+2), truncation=True).to(device)
pro_tokens_val = PLM_tokenizer(val_seqs, return_tensors='pt', padding='max_length', max_length=(max_pro_len_val+2), truncation=True).to(device)

# Make data sets
train_data_set = biosnapdataset(ligand_embs_train, protein_embs_train, np.array(train_labs, dtype=np.int64), lig_tokens_train, pro_tokens_train)
train_dataloader = DataLoader(train_data_set, batch_size=16, shuffle=True) #32 for others
test_data_set = biosnapdataset(ligand_embs_val, protein_embs_val, np.array(val_labs, dtype=np.int64), lig_tokens_val, pro_tokens_val)
test_dataloader = DataLoader(test_data_set, batch_size=16, shuffle=True) #32 for others


# In[ ]:


#@title Train model on protein embeddings only
# Train the model 3 independent times. Report the highest performance on the test set each time.
for n in range(3):

    # Simple model trained on concatenated protein embeddings
    model = torch.nn.Sequential(
        torch.nn.Linear(320, 2),
    )

    # Move to device, define optimizer, loss
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Record the best test set accuracy
    best_test_acc = 0

    epochs_no_improve = 0
    patience = 5

    # 100 training epochs per run
    for epoch in tqdm(range(EPOCHS)):

        # Start training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Load data for batch
        for batch in train_dataloader:
            _, protein_embs, labels, _, _ = batch

            # perform forward pass, compute loss
            protein_embs, labels = protein_embs.to(device), labels.to(device)
            outputs = model(protein_embs)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Get classification performance
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
                _, protein_embs, labels, _, _ = batch

                # Concatenate, forward pass, compute loss
                protein_embs, labels = protein_embs.to(device), labels.to(device)
                outputs = model(protein_embs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)

                # Get classification performance
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            test_loss /= total
            test_accuracy = correct / total

            # Save if model had best accuracy so far
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                #epochs_no_improve = 0
                #torch.save(model.state_dict(), "protein_emb_model.pth")
            #else:
                #epochs_no_improve += 1

            if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

            #if epochs_no_improve >= patience:
                #print(f"Early stopping triggered at epoch {epoch + 1}")
                #break


    # Print the best performance over all 100 epochs
    print(best_test_acc)

# Visualize protein embeddings
embs_cat = protein_embs_val

# Perform PCA
embs_cat = np.array(embs_cat)
tsne = TSNE(n_components=2, perplexity=10, random_state=1)
embs_cat = tsne.fit_transform(embs_cat)

# Colormap
cm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["royalblue","mediumseagreen","crimson"])

# Plot
plt.figure(figsize=(6, 6))
scatter = plt.scatter(embs_cat[:, 0], embs_cat[:, 1], c=[1]*embs_cat.shape[0], cmap=cm, s=10, alpha=0.9)
plt.xlabel("t-SNE-1", fontsize=18)
plt.ylabel("t-SNE-2", fontsize=18)
plt.scatter([], [], c='red', edgecolor='k', alpha=0.7, label='Binding pair')
plt.scatter([], [], c='blue', edgecolor='k', alpha=0.7, label='Non-binding pair')
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('./protein_only.png', dpi=600, bbox_inches='tight')
plt.close()


# In[ ]:


#@title Train model on ligand embeddings only

# Train the model 3 independent times. Report the highest performance on the test set each time.
for n in range(3):

    # Simple model trained on concatenated ligand-protein embeddings
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 2),
    )

    # Move to device, define optimizer, loss
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Record the best test set accuracy
    best_test_acc = 0

    epochs_no_improve = 0
    patience = 5

    # 100 training epochs per run
    for epoch in tqdm(range(EPOCHS)):

        # Start training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Load data for batch
        for batch in train_dataloader:
            ligand_embs, _, labels, _, _ = batch

            # perform forward pass, compute loss
            ligand_embs, labels = ligand_embs.to(device), labels.to(device)
            outputs = model(ligand_embs)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Get classification performance
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
                ligand_embs, _, labels, _, _ = batch

                # forward pass, compute loss
                ligand_embs, labels = ligand_embs.to(device), labels.to(device)
                outputs = model(ligand_embs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)

                # Get classification performance
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            test_loss /= total
            test_accuracy = correct / total

            # Save if model had best accuracy so far
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                #epochs_no_improve = 0

            #else:
                #epochs_no_improve += 1
                #torch.save(model.state_dict(), "ligand_emb_model.pth")
            if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

            #if epochs_no_improve >= patience:
                #print(f"Early stopping triggered at epoch {epoch + 1}")
                #break

    # Print the best performance over all 500 epochs
    print(best_test_acc)

# Visualize ligand embeddings
embs_cat = ligand_embs_val

# Perform PCA
embs_cat = np.array(embs_cat)
tsne = TSNE(n_components=2, perplexity=10, random_state=1)
embs_cat = tsne.fit_transform(embs_cat)

# Colormap
cm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["royalblue","mediumseagreen","crimson"])

# Plot
plt.figure(figsize=(6, 6))
scatter = plt.scatter(embs_cat[:, 0], embs_cat[:, 1], c=[1]*embs_cat.shape[0], cmap=cm, s=10, alpha=0.9)
plt.xlabel("t-SNE-1", fontsize=18)
plt.ylabel("t-SNE-2", fontsize=18)
plt.scatter([], [], c='red', edgecolor='k', alpha=0.7, label='Binding pair')
plt.scatter([], [], c='blue', edgecolor='k', alpha=0.7, label='Non-binding pair')
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('./ligand_only.png', dpi=600, bbox_inches='tight')
plt.close()


# In[ ]:


#@title Train model on concatenated embeddings
# Train the model 3 independent times. Report the highest performance on the test set each time.
for n in range(3):

    # Simple model trained on concatenated protein-ligand embeddings
    model = torch.nn.Sequential(
        torch.nn.Linear(1088, 2),
    )

    # Move to device, define optimizer, loss
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Record the best test set accuracy
    best_test_acc = 0

    epochs_no_improve = 0
    patience = 5

    # 500 training epochs per run
    for epoch in tqdm(range(EPOCHS)):

        # Start training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Load data for batch
        for batch in train_dataloader:
            ligand_embs, protein_embs, labels, _, _ = batch

            # Concatenation occurs here
            inputs = torch.cat((ligand_embs, protein_embs), dim=1)

            # perform forward pass, compute loss
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Get classification performance
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
                ligand_embs, protein_embs, labels, _, _ = batch

                # Concatenate, forward pass, compute loss
                inputs = torch.cat((ligand_embs, protein_embs), dim=1)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)

                # Get classification performance
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            test_loss /= total
            test_accuracy = correct / total

            # Save if model had best accuracy so far
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                #epochs_no_improve = 0
                #torch.save(model.state_dict(), "concat_model.pth")
            #else:
                #epochs_no_improve += 1

            if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

            #if epochs_no_improve >= patience:
                #print(f"Early stopping triggered at epoch {epoch + 1}")
                #break

    # Print the best performance over all 500 epochs
    print(best_test_acc)

# Visualization for concatenation

# Concatenate
embs_cat = np.concatenate((ligand_embs_val, protein_embs_val), axis=1)

# Perform PCA
embs_cat = np.array(embs_cat)
tsne = TSNE(n_components=2, perplexity=10, random_state=1)
embs_cat = tsne.fit_transform(embs_cat)

# Colormap
cm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["royalblue","mediumseagreen","crimson"])

# Plot
plt.figure(figsize=(6, 6))
scatter = plt.scatter(embs_cat[:, 0], embs_cat[:, 1], c=val_labs, cmap=cm, s=10, alpha=0.9)
plt.xlabel("t-SNE-1", fontsize=18)
plt.ylabel("t-SNE-2", fontsize=18)
plt.scatter([], [], c='red', edgecolor='k', alpha=0.7, label='Binding pair')
plt.scatter([], [], c='blue', edgecolor='k', alpha=0.7, label='Non-binding pair')
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('./concatenation.png', dpi=600, bbox_inches='tight')
plt.close()


# In[ ]:


#@title Train contrastive learning model
import torch.nn.functional as F
from torch.nn import CosineSimilarity

# Simple contrastive learning model
class CLModel(nn.Module):
  def __init__(self):
    super(CLModel, self).__init__()

    # Projection network for ligand
    self.ligand_proj = torch.nn.Sequential(
      torch.nn.Linear(768, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 128),
    )

    # Projection network for protein
    self.protein_proj = torch.nn.Sequential(
      torch.nn.Linear(320, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 128),
    )

  def forward(self, lig, pro):
    # Project and normalize
    lig = self.ligand_proj(lig)
    pro = self.protein_proj(pro)
    lig = F.normalize(lig, p=2, dim=1)
    pro = F.normalize(pro, p=2, dim=1)
    return lig, pro

# Train the model 3 independent times. Report the highest performance on the test set each time.
for n in range(3):

    # Define model and other components
    model = CLModel()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CosineEmbeddingLoss()

    best_test_acc = 0

    epochs_no_improve = 0
    patience = 5

    for epoch in tqdm(range(EPOCHS)):

        # Start training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Load data for batch
        for batch in train_dataloader:
            ligand_embs, protein_embs, labels, _, _ = batch

            # Replace zeros with -1s for cosine loss
            labels[labels == 0] = -1

            # Forward pass
            ligand_embs, protein_embs, labels = ligand_embs.to(device), protein_embs.to(device), labels.to(device)
            lig_out, pro_out = model(ligand_embs, protein_embs)

            # Cosine similarity loss
            loss = criterion(lig_out, pro_out, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Get classification metrics via cosine similarity
            train_loss += loss.item() * labels.size(0)
            similarities = CosineSimilarity(dim=1, eps=1e-6)(lig_out, pro_out)
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
                ligand_embs, protein_embs, labels, _, _ = batch

                # Replace zeros with -1s for cosine loss
                labels[labels == 0] = -1

                # Forward pass, Cosine similarity loss
                ligand_embs, protein_embs, labels = ligand_embs.to(device), protein_embs.to(device), labels.to(device)
                lig_out, pro_out = model(ligand_embs, protein_embs)
                loss = criterion(lig_out, pro_out, labels)

                # Get classification metrics by cosine sim
                test_loss += loss.item() * labels.size(0)
                similarities = CosineSimilarity(dim=1, eps=1e-6)(lig_out, pro_out)
                preds = (similarities > 0).long() * 2 - 1 # Map to -1 or 1
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            test_loss /= total
            test_accuracy = correct / total

            # Save if model had best accuracy so far
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                #epochs_no_improve = 0
                torch.save(model.state_dict(), "contrastive_model.pth")
            #else:
                #epochs_no_improve += 1

            if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

            #if epochs_no_improve >= patience:
                #print(f"Early stopping triggered at epoch {epoch + 1}")
                #break

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
    ligand_embs, protein_embs, labels, _, _ = batch
    ligand_embs, protein_embs, labels = ligand_embs.to(device), protein_embs.to(device), labels.to(device)
    lig_out, pro_out = model(ligand_embs, protein_embs)

    for example, y in zip(lig_out, labels):
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


# In[ ]:


#@title Train attention model
EPOCHS=100
# Minimal cross attention network
class AttentionModel(nn.Module):
  def __init__(self, CLM, PLM):
    super(AttentionModel, self).__init__()
    self.CLM = CLM
    self.PLM = PLM
    self.attn = nn.MultiheadAttention(320, 1, batch_first=True)
    self.lig_proj = nn.Linear(768,320) #NOTE: Need to project ligand to same dimensionality
    self.prediction_head = torch.nn.Sequential(
        torch.nn.Linear(320, 2)
    )
    self.post_attn_norm = nn.LayerNorm(320)
    # Freeze the anchor and augment models
    for param in self.CLM.parameters(): param.requires_grad = False
    for param in self.PLM.parameters(): param.requires_grad = False

  def forward(self, lig_tokens, pro_tokens, attn_mask=None):

    # Obtain protein sequence embedding L x D. len(pro_tokens.input_ids) is len(protein_sequence)+2
    with torch.no_grad():
        results = self.PLM.forward(pro_tokens['input_ids'], pro_tokens['attention_mask'], output_hidden_states=True)
    t1 = results.hidden_states[-1]

    # Obtain ligand sequence embedding L x D. len(lig_tokens.input_ids) is len(lig_sequence)+2
    with torch.no_grad():
        results = self.CLM.forward(lig_tokens['input_ids'], lig_tokens['attention_mask'], output_hidden_states=True)
    t2 = results.hidden_states[-1]
    t2 = self.lig_proj(t2) #Project down ligand embeds to same dimensionality for attention

    # Create attn mask
    attn_mask = torch.matmul(
        pro_tokens["attention_mask"].unsqueeze(2).float(),
        lig_tokens['attention_mask'].unsqueeze(1).float(),
    ).repeat(1, 1, 1)

    # Perform attention
    output, attn_weights = self.attn(
        query=t1, key=t2, value=t2, attn_mask=attn_mask, average_attn_weights=False
    )
    output = self.post_attn_norm(output) + t1

    # Take the mean. Exclude padding tokens
    mask_sum = pro_tokens['attention_mask'].sum(dim=1, keepdim=True).clamp(min=1e-6)  # Avoid division by zero
    output = (output * pro_tokens['attention_mask'].unsqueeze(2)).sum(dim=1) / mask_sum

    return self.prediction_head(output), output

for n in range(3):
    # Define model and other components
    model = AttentionModel(CLM, PLM)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_test_acc = 0

    epochs_no_improve = 0
    patience = 5

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_dataloader:
            # Forward pass
            _, _, labels, lig_tokens, pro_tokens = batch
            labels = labels.to(device)
            outputs, _ = model(lig_tokens, pro_tokens)
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
                _, _, labels, lig_tokens, pro_tokens = batch
                labels = labels.to(device)
                outputs, _ = model(lig_tokens, pro_tokens)
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
            #epochs_no_improve = 0
            torch.save(model.state_dict(), "attention_model.pth")
        #else:
            #epochs_no_improve += 1

        if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        #if epochs_no_improve >= patience:
                #print(f"Early stopping triggered at epoch {epoch + 1}")
                #break

    print(best_test_acc)

#@title Visualization for attention
model = AttentionModel(CLM, PLM)
model = model.to(device)
model.load_state_dict(torch.load("./attention_model.pth", weights_only=True))
model.eval()

embs_attention = []
y_true = []
with torch.no_grad():
    for batch in test_dataloader:
        _, _, labels, lig_tokens, pro_tokens = batch
        labels = labels.to(device)
        lig_out, _ = model(lig_tokens, pro_tokens)
        for example, y in zip(lig_out, labels):
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


# In[ ]:

#@title Train composition style model
EPOCHS=100
# Minimal composition of language models
target_layers = [3,4,5] #0,3,5
clm_layers = [9,10,11] #3,7,11
class CompositionModel(nn.Module):
  def __init__(self, CLM, PLM):
    super(CompositionModel, self).__init__()
    self.CLM = CLM
    self.PLM = PLM
    self.cross_atten_layers = nn.ModuleList([
        nn.MultiheadAttention(320, 20, batch_first=True) for i in range(len(target_layers)) #NOTE: Trying 4 heads or 8 next.
    ])
    self.post_attn_norms = nn.ModuleList([
        nn.LayerNorm(320) for i in range(len(target_layers))
    ])
    self.prediction_head = torch.nn.Sequential(
        torch.nn.Linear(320, 2),
    )
    self.projection_chems = nn.ModuleList([
        nn.Linear(768,320) for i in range(len(target_layers))
    ])

    #self.trainable_norm = nn.LayerNorm(320, elementwise_affine=True)
    self.esm_layers = 6
    # Freeze the anchor and augment models
    for param in self.CLM.parameters(): param.requires_grad = False
    for param in self.PLM.parameters(): param.requires_grad = False

  def forward(self, lig_input, prot_input, attn_mask=None):
    # Create attn mask
    attn_mask = torch.matmul(
        prot_input["attention_mask"].unsqueeze(2).float(),
        lig_input['attention_mask'].unsqueeze(1).float(),
    ).repeat(20, 1, 1)

    # Create the esm attention masks correctly
    prot_attn_mask = self.PLM.get_extended_attention_mask(prot_input['attention_mask'], prot_input["input_ids"].size())

    # Embedding layer
    pro = self.PLM.esm.embeddings(prot_input["input_ids"], prot_input["attention_mask"]).to(device)
    with torch.no_grad():

        outputs = self.CLM.molformer(
        input_ids=lig_input["input_ids"],
        attention_mask=lig_input["attention_mask"],
        output_hidden_states=True)
        lig = outputs.hidden_states[0].to(device)  # or select another layer as needed

    # Layerwise forward pass
    counter = 0
    for i in range(0, self.esm_layers):

        # Update embeddings
        pro = self.PLM.esm.encoder.layer[i](pro, prot_attn_mask)[0]#, prot_input["attention_mask"][:, None, None, :])[0]

        if i in target_layers:
            # Perform cross attn and layer norm
            lig = outputs.hidden_states[clm_layers[counter]].to(device) #Get the specific layer we desire for augment and set to device
            lig_proj = self.projection_chems[counter](lig) #Project down the ligand to the correct dimensionality
            attn_out, comp_attn_weights = self.cross_atten_layers[counter](query=pro, key=lig_proj, value=lig_proj, attn_mask=attn_mask, average_attn_weights=False)
            attn_out = self.post_attn_norms[counter](attn_out)
            pro = pro + attn_out
            counter += 1
    
    # Take the normal mean pool
    pro = self.PLM.esm.encoder.emb_layer_norm_after(pro)
    #pro = self.trainable_norm(pro)
    mask_sum = prot_input['attention_mask'].sum(dim=1, keepdim=True).clamp(min=1e-6)  # Avoid division by zero
    pro = (pro * prot_input['attention_mask'].unsqueeze(2)).sum(dim=1) / mask_sum
    return self.prediction_head(pro), pro
  

# NOTE: Test the reduced VRAM
# Freeze the anchor and augment models
#for param in CLM.parameters(): param.requires_grad = False
#for param in PLM.parameters(): param.requires_grad = False

for n in range(3):

    # Define model and other components
    model = CompositionModel(CLM, PLM)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_test_acc = 0

    epochs_no_improve = 0
    patience = 5

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_dataloader:
            # Forward pass
            _, _, labels, lig_tokens, pro_tokens = batch
            labels = labels.to(device)
            outputs, _ = model(lig_tokens, pro_tokens)
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
                _, _, labels, lig_tokens, pro_tokens = batch
                labels = labels.to(device)
                outputs, _ = model(lig_tokens, pro_tokens)
                loss = criterion(outputs, labels)
                # Compute classification metrics
                test_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        test_loss /= total
        test_accuracy = correct / total

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            torch.save(model.state_dict(), "comp_big_model.pth")
            #epochs_no_improve = 0
        #else:
            #epochs_no_improve += 1

        if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        #if epochs_no_improve >= patience:
                #print(f"Early stopping triggered at epoch {epoch + 1}")
                #break

    print(best_test_acc)

#@title Visualization for composition
model = CompositionModel(CLM, PLM)
model = model.to(device)
model.load_state_dict(torch.load("./comp_big_model.pth", weights_only=True))
model.eval()

embs_attention = []
y_true = []
with torch.no_grad():
    for batch in test_dataloader:
        _, _, labels, lig_tokens, pro_tokens = batch
        labels = labels.to(device)
        lig_out, _ = model(lig_tokens, pro_tokens)
        for example, y in zip(lig_out, labels):
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
plt.savefig('./composition.png', dpi=600, bbox_inches='tight')
plt.close()

