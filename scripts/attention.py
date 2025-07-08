
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
