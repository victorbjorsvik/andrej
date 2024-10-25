import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
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

class Head(nn.Module):
  
  def __init__(self, head_size):
    super().__init__()
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    B,T,C = x.shape
    q = self.query(x) # (B,T,C)
    k = self.key(x)   # (B,T,C)
    # compute attention scores ('affinities)
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,16,C) --->  (B,T,T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
    wei = F.softmax(wei, dim=-1) # (B,T,T)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    v = self.value(x) # (B,T,C)
    out = wei @ v # (B,T,T) @ (B,T,C) ---> (B,T,C)
    return out
  
class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in paralell """
  
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)


  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  """ a simple linear layer followed by a non-linearity """

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4*n_embd),
      nn.ReLU(),
      nn.Linear(4*n_embd, n_embd), # projection-layer
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)
   
class Block(nn.Module):
  """ Transformer block: communication followed by computation """
  def __init__(self, n_embd, n_head):
    # n_embd: embedding dimension, n_head: the number heads we'd like
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x)) # Using the pre-norm formulation (deviates from attention is all you need)
    x = x + self.ffwd(self.ln2(x))
    return x

# Basic Bitch Bigram
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # Final layer norm before decoding
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B,T = idx.shape

    # idx and targets are both (B,T) tensors of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_embedding = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
    x = tok_emb + pos_embedding # (B,T,C)
    x = self.blocks(x) # apply several blocks multiheaded self-attention and MLP . (B,T,C)
    x = self.ln_f(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T,vocab_size)

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
      # crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # Get predictions
      logits, loss = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # Becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=1) # (B,C)
      # Sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
      # append sampled index to running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
  
model = BigramLanguageModel()
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