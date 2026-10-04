import torch

from rewind.model import SeqConfig, SequenceCTR

CFG = SeqConfig(n_items=1000, max_len=8, d_model=16, n_heads=2, n_layers=2)


def _model():
    torch.manual_seed(0)
    return SequenceCTR(CFG).eval()


def test_output_is_one_logit_per_example():
    m = _model()
    items = torch.randint(1, 1000, (5, 8))
    actions = torch.randint(1, 5, (5, 8))
    target = torch.randint(1, 1000, (5,))
    out = m(items, actions, target)
    assert out.shape == (5,)
    assert torch.isfinite(out).all()


def test_padding_does_not_change_the_prediction():
    # The same 3-step history, once with 5 pads in front (L=8) and once with no pads (L=3).
    # Padding means "nothing happened", so the prediction must be identical. Because history is
    # left-padded, the short input's tokens sit at positions 5..8 of the long one; _with_offset
    # gives them those same position IDs so the two runs are directly comparable.
    m = _model()
    items = torch.tensor([[11, 22, 33]])
    actions = torch.tensor([[1, 2, 1]])
    target = torch.tensor([44])
    pads = torch.zeros(1, 5, dtype=torch.long)

    padded = m(torch.cat([pads, items], dim=1), torch.cat([pads, actions], dim=1), target)
    unpadded = _with_offset(m, items, actions, target, offset=5)
    torch.testing.assert_close(padded, unpadded)


def test_all_padding_history_gives_finite_output():
    # A user with no history at all must still get a valid prediction, not NaN.
    m = _model()
    out = m(torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, 8, dtype=torch.long), torch.tensor([5, 6]))
    assert torch.isfinite(out).all()


def test_changing_an_old_action_changes_the_prediction():
    # Sanity check that history is actually used: if the prediction ignored history, a model
    # could never beat a tabular baseline and the comparison would be meaningless.
    m = _model()
    items = torch.randint(1, 1000, (1, 8))
    actions = torch.ones(1, 8, dtype=torch.long)
    target = torch.tensor([7])
    before = m(items, actions, target)
    items[0, 0] = (items[0, 0] % 999) + 1
    after = m(items, actions, target)
    assert not torch.allclose(before, after)


def _with_offset(m, items, actions, target, offset):
    """Run the model on an unpadded history but with position IDs starting at `offset`."""
    original = m.pos_emb.forward
    m.pos_emb.forward = lambda idx: original(idx + offset)
    try:
        return m(items, actions, target)
    finally:
        m.pos_emb.forward = original
