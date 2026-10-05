"""Phase 3: the GPU data feeder and the sequence model, on a tiny log with known answers."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rewind.baselines.models import DCN
from rewind.data.sequences import PROFILE_COLS, save
from rewind.features.tabular import build_features
from rewind.sequence.data import GAP_EDGES, TOKEN_FIELDS, SequenceData
from rewind.sequence.model import SequenceModel, SequenceModelConfig, param_counts, size_head
from tests.test_features import random_log

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Fake log -> Phase 1 sequences -> Phase 2 features -> feeder, all on disk like the real run."""
    root = tmp_path_factory.mktemp("pipe")
    cols = random_log(seed=7, n=400)
    save(cols, root / "seq")
    build_features(root / "seq", root / "feat")
    return cols, SequenceData(CPU, root / "seq", root / "feat")


def brute_history(cols, i: int, length: int):
    """The user's impressions from strictly earlier seconds, newest last, the last `length` of them."""
    j = np.flatnonzero((cols["user"] == cols["user"][i]) & (cols["time_stamp"] < cols["time_stamp"][i]))
    return j[-length:]  # rows are stored sorted by (user, time), so this is chronological


def test_feeder_matches_brute_force(pipeline):
    cols, data = pipeline
    length = 8
    rows = np.arange(len(cols["user"]))
    b = data.batch(torch.from_numpy(rows), length, trim=False)
    tokens = np.stack([cols[f] for f in TOKEN_FIELDS], axis=1)
    for i in rows:
        h = brute_history(cols, i, length)
        pad = length - len(h)
        real = b["real"][i].numpy()
        assert real.tolist() == [False] * pad + [True] * len(h), i
        np.testing.assert_array_equal(b["hist_tokens"][i, pad:].numpy(), tokens[h], err_msg=str(i))
        assert (b["hist_tokens"][i, :pad] == 0).all()
        np.testing.assert_array_equal(b["hist_action"][i, pad:].numpy(), cols["clk"][h] + 1)
        gap = cols["time_stamp"][i] - cols["time_stamp"][h]
        np.testing.assert_array_equal(b["hist_gap"][i, pad:].numpy(),
                                      np.searchsorted(GAP_EDGES, gap, side="right") + 1)
        np.testing.assert_array_equal(b["cand_tokens"][i].numpy(), tokens[i])
        assert b["label"][i].item() == cols["clk"][i]


def test_own_label_never_reaches_the_inputs(pipeline):
    cols, data = pipeline
    i = int(np.flatnonzero(data.prior > 3)[0])
    rows = torch.tensor([i])
    before = data.batch(rows, 8, trim=False)
    data.clk[i] = 1 - data.clk[i]  # flip this question's own answer
    try:
        after = data.batch(rows, 8, trim=False)
    finally:
        data.clk[i] = 1 - data.clk[i]
    for k in before:
        if k == "label":
            assert before[k].item() != after[k].item()
        else:
            assert torch.equal(before[k], after[k]), f"{k} changed when only the label changed"


@pytest.mark.parametrize("gap, bucket", [(1, 1), (59, 1), (60, 2), (299, 2), (300, 3), (3599, 5), (3600, 6),
                                         (86399, 10), (86400, 11), (345599, 12), (345600, 13), (9_999_999, 13)])
def test_time_gap_bucket_edges(gap, bucket):
    edges = torch.tensor(GAP_EDGES, dtype=torch.int32)
    assert torch.bucketize(torch.tensor([gap], dtype=torch.int32), edges, right=True).item() + 1 == bucket


def small_model(data, length=16, **kw):
    torch.manual_seed(0)
    cfg = SequenceModelConfig(cardinalities=data.cardinalities, n_context=len(data.context_names),
                              d_model=32, n_heads=2, n_layers=2, head_hidden=32, max_len=length, **kw)
    return SequenceModel(cfg).eval()


def test_padding_and_trimming_change_no_prediction(pipeline):
    cols, data = pipeline
    model = small_model(data, length=16)
    short = np.flatnonzero(data.prior < 8)  # these histories fit in 8, so 8 and 16 differ only by padding
    rows = torch.from_numpy(short)
    with torch.no_grad():
        p8 = model(data.batch(rows, 8, trim=False))
        p16 = model(data.batch(rows, 16, trim=False))
        p16_trim = model(data.batch(rows, 16, trim=True))
    torch.testing.assert_close(p8, p16)
    torch.testing.assert_close(p16, p16_trim)


def test_history_actually_changes_predictions(pipeline):
    cols, data = pipeline
    model = small_model(data)
    i = int(np.flatnonzero(data.prior > 3)[0])
    b = data.batch(torch.tensor([i]), 16)
    with torch.no_grad():
        before = model(b)
        b["hist_action"][0, -1] = 3 - b["hist_action"][0, -1]  # newest item: clicked <-> not clicked
        after = model(b)
    assert not torch.allclose(before, after)


def test_model_learns_a_pattern_that_lives_only_in_the_history():
    # Label = "was the newest history item clicked?" Nothing else carries it, so the model can
    # only learn it by reading the sequence through attention.
    torch.manual_seed(0)
    B, L = 512, 6
    cards = {f: 20 for f in TOKEN_FIELDS} | {f: 5 for f in PROFILE_COLS}
    model = SequenceModel(SequenceModelConfig(cardinalities=cards, n_context=3, d_model=32, n_heads=2,
                                              n_layers=2, head_hidden=32, max_len=L))
    action = torch.randint(1, 3, (B, L))
    b = {
        "hist_tokens": torch.randint(1, 20, (B, L, 6)), "hist_action": action,
        "hist_gap": torch.randint(1, 14, (B, L)), "real": torch.ones(B, L, dtype=torch.bool),
        "cand_tokens": torch.randint(1, 20, (B, 6)), "profile": torch.randint(1, 5, (B, len(PROFILE_COLS))),
        "context": torch.randn(B, 3), "label": (action[:, -1] == 2).float(),
    }
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(model(b), b["label"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"could not learn a history-only pattern: loss {loss.item():.3f}"


def test_embedding_tables_equal_dcn_and_dense_size_is_matched(pipeline):
    _, data = pipeline
    cards = data.cardinalities
    dcn = DCN(list(cards.values()), n_num=25, dim=16)
    dcn_tables = sum(e.weight.numel() for e in dcn.emb.cat)
    target = 1_364_049
    cfg = size_head(SequenceModelConfig(cardinalities=cards, n_context=len(data.context_names)), target)
    counts = param_counts(SequenceModel(cfg))
    assert counts["id_embedding"] == dcn_tables
    assert abs(counts["dense"] / target - 1) < 0.01


def test_user_fraction_keeps_whole_users_and_valid_histories(tmp_path):
    cols = random_log(seed=7, n=400)
    save(cols, tmp_path / "seq")
    build_features(tmp_path / "seq", tmp_path / "feat")
    full = SequenceData(CPU, tmp_path / "seq", tmp_path / "feat")
    part = SequenceData(CPU, tmp_path / "seq", tmp_path / "feat", user_fraction=0.5)
    n = len(part.user)
    assert 0 < n < len(full.user)
    assert full.user[n] != full.user[n - 1]  # the cut falls exactly between two users
    rows = torch.arange(n)
    a, b = full.batch(rows, 8, trim=False), part.batch(rows, 8, trim=False)
    for k in a:
        assert torch.equal(a[k], b[k]), k  # identical inputs for every row that was kept
