"""验证 EMA、学习率端点和随机状态恢复，防止整理代码后改变训练协议。"""
import random

import numpy as np
import pytest
import torch
from torch import nn

from seqeft.train import TrainableEMA, build_scheduler
from seqeft.utils import restore_rng, rng_state


def test_ema_starts_after_first_update_and_restores():
    model = nn.Linear(1, 1, bias=False)
    model.register_parameter("frozen", nn.Parameter(torch.ones(1), requires_grad=False))
    ema = TrainableEMA(model, decay=0.99)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update()
    assert set(ema.shadow) == {"weight"}
    assert ema.shadow["weight"].item() == 2.0
    with torch.no_grad():
        model.weight.fill_(4.0)
    ema.update()
    assert ema.shadow["weight"].item() == pytest.approx(2.02)
    with pytest.raises(RuntimeError):
        with ema.applied():
            assert model.weight.item() == pytest.approx(2.02)
            raise RuntimeError("模拟评价异常")
    assert model.weight.item() == 4.0
    assert ema.num_updates == 2


def test_reference_learning_rate_endpoints():
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_scheduler(optimizer, {"training": {"warmup_epochs": 5, "epochs": 150}})
    used = []
    for _ in range(150):
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert used[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert used[5] == 1.0
    assert used[-1] == pytest.approx(0.0)
    assert all(a >= b for a, b in zip(used[5:-1], used[6:]))


def test_rng_round_trip():
    generator = torch.Generator().manual_seed(42)
    state = rng_state(generator)
    expected = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator))
    restore_rng(state, generator)
    actual = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
