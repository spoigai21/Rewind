import torch

from rewind.baselines.models import MODELS, DeepFM


def test_fm_shortcut_equals_sum_over_pairs():
    torch.manual_seed(0)
    m = DeepFM([10, 10, 10], n_num=2, dim=4)
    cat, num = torch.randint(0, 10, (6, 3)), torch.randn(6, 2)
    e = m.emb(cat, num)
    fast = 0.5 * (e.sum(1).pow(2) - e.pow(2).sum(1)).sum(-1)
    slow = sum((e[:, i] * e[:, j]).sum(-1) for i in range(e.shape[1]) for j in range(i + 1, e.shape[1]))
    torch.testing.assert_close(fast, slow)


def test_every_model_outputs_one_finite_logit_and_learns():
    # A tiny problem where the label is a function of one categorical field: every model must fit
    # it, which shows gradients reach the embeddings.
    torch.manual_seed(0)
    cat = torch.randint(0, 4, (512, 2))
    num = torch.randn(512, 3)
    y = (cat[:, 0] >= 2).float()
    for name, M in MODELS.items():
        m = M([4, 4], 3, dim=4)
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        for _ in range(200):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(cat, num), y)
            opt.zero_grad(); loss.backward(); opt.step()
        out = m(cat, num)
        assert out.shape == (512,) and torch.isfinite(out).all(), name
        assert loss.item() < 0.1, f"{name} failed to fit a trivial pattern: loss {loss.item():.3f}"
