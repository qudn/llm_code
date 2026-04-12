import torch
def stable_cross_entropy(logits, target):
  log_softmax = torch.logsumexp(logits, dim = -1, keepdim = True)
  loss = -log_softmax[range(len(target)), target]
  return loss.mean()
