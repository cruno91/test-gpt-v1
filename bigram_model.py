import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if Metal is available.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(x)
else:
    device = torch.device("cpu")
    print("MPS device not found.")

# Open the text file.
with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# Get the set of unique characters in the text.
chars = sorted(set(text))
print(chars)

# Create a tokenizer to convert between characters and numerical indices via an encoder and a decoder.
# Encoder and a decoder:
# Encoder: converts a string to an integer.
strings_to_ints = {c: i for i, c in enumerate(chars)}
encode = lambda s: [strings_to_ints[c] for c in s]
# Decoder: converts an integer to a string.
ints_to_strings = {i: c for i, c in enumerate(chars)}
decode = lambda x: ''.join([ints_to_strings[i] for i in x])

# Convert the text to integers.
# encoded = encode("Hello")
# print(encoded)
# decoded = decode(encoded)
# print(decoded)

# Convert the text to integers.
# dtype = torch.long: 64-bit integer (signed) - Important for pytorch to know the type of data.
data = torch.tensor(encode(text), dtype=torch.long)

# You have to split your training data corpus into chunks so that you create output like the original, but not copies
# of the original.
# The chunks are also needed to be able to scale the model.
# It's called the bi-gram language model.
block_size = 8
# Number of blocks you can do in parallel.
batch_size = 4
# Number of iterations.
max_iters = 10000
# Learning rate.
learning_rate = 3e-4 # 0.0003
# Eval iters
eval_iters = 250
# Dropout rate.
# Take 20% of the neurons and turn them off.
dropout = 0.2

# Get the training and validation splits.
n = int(0.8*len(data))
train_data, val_data = data[:n], data[n:]

# Get a batch of data.
def get_batch(split):
    data = train_data if split == "train" else val_data
    # Take a random integer between 0 and the length of the data minus the block size.
    # So if you get the index that's at the length of the n minus block size you'll still have a block size of 8.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    # Get the data from the random index to the random index plus the block size.
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Get the data from the random index plus 1 to the random index plus the block size plus 1.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # Move the data to the device.
    x, y = x.to(device), y.to(device)
    return x, y

x, y = get_batch("train")
print('inputs:')
print(x)
print('targets:')
print(y)

# Estimate the loss.
# torch.no_grade() is a decorator that disables gradient calculation.
# This is useful for inference because you don't need to calculate gradients.
# It also reduces memory consumption for computations that would otherwise have requires_grad=True.
@torch.no_grad()
def estimate_loss():
    out = {}
    # Dropout is turned off in evaluation.
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Bigram language model.
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print("when input is", context, "target is", target)

# Create the model.
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Embedding layer: Converts an integer to a vector.
        # Lookup table tokens in a block.  Each token is a number with a probability distribution
        # across the vocabulary to predict the next token.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets=None):
        # Logits are the output of the model before the softmax activation function.
        # The logits are the raw values that are passed to the activation function.
        # The activation function then normalizes the logits and converts them to a probability distribution.
        # Logits are the unnormalized output of a neural network.
        # Layman: Logits are a bunch of normalized floating point numbers.
        # Say you have [2, 4, 6] and you want to normalize it.
        # You would divide each number by the sum of the numbers.
        # 2 / (2 + 4 + 6) = 0.1
        # 4 / (2 + 4 + 6) = 0.2
        # 6 / (2 + 4 + 6) = 0.3
        # The sum of the numbers is 1.
        # This becomes [0.1, 0.2, 0.3].
        # Say those equate to the probabilities of a, b, and c.
        # The probability of a is 0.1.
        # The probability of b is 0.2.
        # The probability of c is 0.3.
        logits = self.token_embedding_table(index)

        # Because targets is none, the loss is none and the code does not execute.
        # Just use the logits which are three-dimensional.
        if targets is None:
            loss = None
        else:
            # batch, time
            # (sequence of integers, we don't know next token, because some we don't know yet),
            # channels (vocab size)
            # Shape is BxTxC.
            B, T, C = logits.shape
            # Because we're paying attention to the vocabulary (channels), the batch
            # and time dimensions are combined.
            # As long as the logits and the targets are the same batch and time we should be alright.
            # PyTorch expects the logits to be a 2D tensor and the targets to be a 1D tensor which is why we use
            # view() to reshape the tensors.
            # Making the first parameter a single parameter of batch by time.
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # cross_entropy is way of measuring loss.
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        # Index is (Batch, Time) array of indicies in the current context.
        for _ in range(max_new_tokens):
            # Get the logits (predictions).
            logits, loss = self.forward(index)
            # Focus only on the last time step.
            # Only care about single previous character because it's a bigram model.  No context before.
            logits = logits[:, -1, :]  # Becomes (B, C)
            # Apply softmax to get the probabilities.
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the probability distribution.
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence.
            # We have a time dimension with 1 element in the 0th position,
            # so when we generate a token we need to add 1 to that position.
            # So we have 0 then we have 1, then 1 and we have 2.  Keep concatenating more tokens onto it.
            index = torch.cat([index, index_next], dim=1)  # (B, T+1)
        return index


vocab_size = len(chars)

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# toch.long is int64 (8 bytes).
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)

# Create the optimizer.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model.
# Standard training loop for basic models.
for i in range(max_iters):
    if i % eval_iters == 0:
        losses = estimate_loss()
        print(f'step: {i}, train loss: {losses["train"]:.3f}, val loss: {losses["val"]:.3f}')
    # Get the batch.
    xb, yb = get_batch("train")
    # Get the logits and the loss.
    logits, loss = model.forward(xb, yb)
    # Zero out the gradients.
    # Set the gradient to none because it occupies less space than default of 0 (int64).
    # Usually only want it on if you are doing large recurrent neural nets which need to
    # understand previous context.
    # If it's 0 it averages the gradients (gradient accumulation).
    optimizer.zero_grad(set_to_none=True)
    # Backpropagate the loss.
    loss.backward()
    # Update the weights.
    optimizer.step()
    # Print the loss.
print(loss.item())
