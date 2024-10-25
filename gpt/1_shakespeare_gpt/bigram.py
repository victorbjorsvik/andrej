import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ---------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All unique characters that appear in text (vocab)
chars = list(sorted(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string and output list of integers
decode = lambda l: "".join([itos[i] for i in l])

# Let's encode (tokenize) and split the entire dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train = data[:n]
val = data[n:]

# Create a dataloader
def get_batch(split):
    """Generate a small batch of data with inputs:x and outputs:y"""
    data = train if split == "train" else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

xb, yb = get_batch("train")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Basic Bitch Bigram
class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):

    # idx and targets are both (B,T) tensors of integers
    logits = self.token_embedding_table(idx) # (B,T,C)

    if targets is None:
        loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B,T) array of indicies in the current context
    for _ in range(max_new_tokens):
      # Get predictions
      logits, loss = self(idx)
      # focus only on the last time step
      logits = logits[:, -1, :] # Becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=1) # (B,C)
      # Sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
      # append sampled index to running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
  
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
   
   # every once in a while evaluate train/val-loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step: {iter}, Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}") 

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))