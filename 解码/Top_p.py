import torch

def top_p(logits, p = 0.9, temperature = 1.0):
  probs = torch.softmax(logits / temperature, dim = -1)

  sorted_probs, sorted_idx = torch.sort(probs, descending = True)

  cumulative_probs = torch.cumsum(sorted_probs, dim = -1)

  mask = cumulative_probs - sorted_probs >= p
  sorted_probs[mask] = 0.0

  sorted_probs /= sorted_probs.sum()

  sampled_idx = torch.multinomial(sorted_probs, num_samples = 1)
  token_idx = sorted_idx[sampled_idx]

  return token_idx.item()
  
