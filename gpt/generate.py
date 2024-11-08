

# prefix tokens
model.eval()
num_return_sequences = 4
max_length = 32
tokens = enc.encode("Hello, I am a large language model,") 
tokens = torch.tensor(tokens, dtype=torch.long) # (9,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 9)
xgen = tokens.to(device)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42 + ddp_rank)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take logits from last position
        logits = logits[:, -1, :]
        # get the probabilites
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # Select a token from the top-k probabilites
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)