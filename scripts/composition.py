LR = 1e-3
EPOCHS = 100
VERBOSE = True

#@title Load mhc data
import pandas as pd

# Load data frames
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

# print(len(mhc_train_pep))
# print(len(mhc_train_rec))
# print(len(mhc_train_lab))
# print(sum(mhc_train_lab)/len(mhc_train_lab))
# print(len(mhc_val_pep))
# print(len(mhc_val_rec))
# print(len(mhc_val_lab))
# print(sum(mhc_val_lab)/len(mhc_val_lab))

#@title Load language models
import math
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoModelForMaskedLM, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

esm_layers = 6
esm_params = 8

# Load pLM and PLM models
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
    
#@title Extract sequence embeddings
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
color_labels = kmeans.fit(peptide_embs_val).labels_

#@title Composition
print('COMPOSITION')

# Minimal composition of language models
target_layers = [0, 3, 5]
class CompositionModel(nn.Module):
  def __init__(self, plm, PLM):
    super(CompositionModel, self).__init__()
    self.plm = plm
    self.PLM = PLM
    self.cross_atten_layers = nn.ModuleList([
        nn.MultiheadAttention(320, 20, batch_first=True) for i in range(len(target_layers))
    ])
    self.post_attn_norms = nn.ModuleList([
        nn.LayerNorm(320) for i in range(len(target_layers))
    ])
    self.prediction_head = torch.nn.Sequential(
        torch.nn.Linear(320, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
        torch.nn.Softmax(dim=1)
    )
    self.esm_layers = 6
    # Freeze the anchor and augment models
    for param in self.plm.parameters(): param.requires_grad = False
    for param in self.PLM.parameters(): param.requires_grad = False

  def forward(self, pep_input, prot_input, attn_mask=None):
    # Create attn mask
    attn_mask = torch.matmul(
        prot_input["attention_mask"].unsqueeze(2).float(),
        pep_input['attention_mask'].unsqueeze(1).float(),
    ).repeat(20, 1, 1)

    # Create the esm attention masks correctly
    pep_attn_mask = self.plm.get_extended_attention_mask(pep_input['attention_mask'], pep_input["input_ids"].size())
    prot_attn_mask = self.PLM.get_extended_attention_mask(prot_input['attention_mask'], prot_input["input_ids"].size())

    # Embedding layer
    pro = self.PLM.esm.embeddings(prot_input["input_ids"], prot_input["attention_mask"]).to(device)
    with torch.no_grad():
        pep = self.plm.esm.embeddings(pep_input["input_ids"], pep_input["attention_mask"]).to(device)

    # Layerwise forward pass
    counter = 0
    for i in range(0, self.esm_layers):

        # Update embeddings
        pro = self.PLM.esm.encoder.layer[i](pro, prot_attn_mask)[0]#, prot_input["attention_mask"][:, None, None, :])[0]
        with torch.no_grad():
            pep = self.plm.esm.encoder.layer[i](pep, pep_attn_mask)[0]#, pep_input["attention_mask"][:, None, None, :])[0]

        if i in target_layers:
            # Perform cross attn and layer norm
            attn_out, _ = self.cross_atten_layers[counter](query=pro, key=pep, value=pep, attn_mask=attn_mask, average_attn_weights=False)
            attn_out = self.post_attn_norms[counter](attn_out)
            pro = pro + attn_out
            counter += 1
    
    pro = self.PLM.esm.encoder.emb_layer_norm_after(pro)

    # Take the mean. Exclude padding tokens and bos/eos
    mask_sum = prot_input['attention_mask'].sum(dim=1, keepdim=True).clamp(min=1e-6)  # Avoid division by zero
    pro = (pro * prot_input['attention_mask'].unsqueeze(2)).sum(dim=1) / mask_sum
    return self.prediction_head(pro), pro

for n in range(3):

    # Define model and other components
    model = CompositionModel(plm, PLM)
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

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            #torch.save(model.state_dict(), "comp_model.pth")
        if VERBOSE: print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    print(best_test_acc)

# #@title Visualization for composition
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import matplotlib
# import matplotlib.pyplot as plt

# model = CompositionModel(plm, PLM)
# model = model.to(device)
# model.load_state_dict(torch.load("./comp_model.pth", weights_only=True))
# model.eval()

# embs_attention = []
# y_true = []
# proteins = []
# with torch.no_grad():
#     for batch in test_dataloader:
#         _, _, labels, pep_tokens, pro_tokens = batch
#         labels = labels.to(device)
#         pep_out, _ = model(pep_tokens, pro_tokens)
#         for example, protein_token, y in zip(pep_out, pro_tokens['input_ids'], labels):
#             embs_attention.append(example.detach().cpu().numpy())
#             y_true.append(y.item())
#             proteins.append(str(tokenizer.decode(protein_token)).replace(' ', '').replace('<CLS>', '').replace('<EOS>',''))

# print(len(set(mhc_val_rec)))
# print(len(set(proteins)))

# # Perform PCA
# embs_attention = np.array(embs_attention)
# tsne = TSNE(n_components=2, perplexity=10, random_state=1)
# embs_attention = tsne.fit_transform(embs_attention)

# # Colormap
# # cm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["royalblue","crimson"])
# import matplotlib.colors as mcolors
# n_colors = len(set(mhc_val_rec))
# colors = plt.cm.viridis(np.linspace(0, 1, n_colors))  # Use any base colormap
# cm = mcolors.ListedColormap(colors)

# # Number the unqiue proteins for labels
# sequence_to_number = {}
# numbered_sequences = []

# for seq in proteins:
#     if seq not in sequence_to_number:
#         sequence_to_number[seq] = len(sequence_to_number)
#     numbered_sequences.append(sequence_to_number[seq])
# print(max(numbered_sequences))

# import math

# import numpy as np
# from matplotlib.colors import ListedColormap
# from matplotlib.cm import hsv


# def generate_colormap(number_of_distinct_colors: int = 80):
#     if number_of_distinct_colors == 0:
#         number_of_distinct_colors = 80

#     number_of_shades = 7
#     number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

#     # Create an array with uniformly drawn floats taken from <0, 1) partition
#     linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

#     # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
#     #     but each saw tooth is slightly higher than the one before
#     # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
#     arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

#     # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
#     arr_by_shade_columns = arr_by_shade_rows.T

#     # Keep number of saw teeth for later
#     number_of_partitions = arr_by_shade_columns.shape[0]

#     # Flatten the above matrix - join each row into single array
#     nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

#     # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
#     initial_cm = hsv(nums_distributed_like_rising_saw)

#     lower_partitions_half = number_of_partitions // 2
#     upper_partitions_half = number_of_partitions - lower_partitions_half

#     # Modify lower half in such way that colours towards beginning of partition are darker
#     # First colours are affected more, colours closer to the middle are affected less
#     lower_half = lower_partitions_half * number_of_shades
#     for i in range(3):
#         initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

#     # Modify second half in such way that colours towards end of partition are less intense and brighter
#     # Colours closer to the middle are affected less, colours closer to the end are affected more
#     for i in range(3):
#         for j in range(upper_partitions_half):
#             modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
#             modifier = j * modifier / upper_partitions_half
#             initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

#     return ListedColormap(initial_cm)

# # numbered_sequences = np.array(numbered_sequences)
# # embs_attention = embs_attention[numbered_sequences > 32]
# # numbered_sequences = numbered_sequences[numbered_sequences > 32]

# # Plot
# plt.figure(figsize=(6, 6))
# scatter = plt.scatter(embs_attention[:, 0], embs_attention[:, 1], c=color_labels, s=10, alpha=0.9, cmap=generate_colormap(8))
# plt.xlabel("t-SNE-1", fontsize=18)
# plt.ylabel("t-SNE-2", fontsize=18)
# plt.scatter([], [], c='crimson', edgecolor='k', alpha=0.7, label='Binding pair')
# plt.scatter([], [], c='royalblue', edgecolor='k', alpha=0.7, label='Non-binding pair')
# plt.scatter([], [], c='mediumseagreen', edgecolor='k', alpha=0.7, label='Non-binding pair')
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.savefig('./composition_byprotein.png', dpi=600, bbox_inches='tight')
# plt.close()
