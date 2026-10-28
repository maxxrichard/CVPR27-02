import torch


def accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return float((probs.argmax(1).cpu() == labels.cpu()).float().mean() * 100.0)


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(1)
    conf, pred, labels = conf.cpu(), pred.cpu(), labels.cpu()
    correct = (pred == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(())
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()
    return float(ece * 100.0)


def negative_log_likelihood(probs: torch.Tensor, labels: torch.Tensor) -> float:
    p = probs.cpu().gather(1, labels.cpu().view(-1, 1)).clamp_min(1e-12)
    return float(-torch.log(p).mean())


def per_class_accuracy(probs: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    pred = probs.argmax(1).cpu(); labels = labels.cpu()
    out = torch.full((K,), float("nan"))
    for k in range(K):
        m = labels == k
        if m.any():
            out[k] = (pred[m] == k).float().mean()
    return out
