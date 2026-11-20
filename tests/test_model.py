"""在真实参考尺寸下验证科学计算合同，不依赖私有数据或权重。"""

import pytest
import torch

from seqeft.model import build_model
from seqeft.smoke import (
    check_alignment,
    check_eeg_branches,
    check_parameter_counts,
    check_vae,
)


@pytest.fixture(scope="module")
def reference_model():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = build_model({}, load_pretrained=False)
    yield model
    torch.set_num_threads(previous_threads)


def test_stage1_forward_backward(reference_model):
    del reference_model
    check_vae()


def test_reference_architecture_and_initialization(reference_model):
    check_parameter_counts(reference_model)


def test_real_eeg_branches_single_frame(reference_model):
    check_eeg_branches(reference_model)


def test_alignment_backpropagates_through_frozen_decoder(reference_model):
    check_alignment(reference_model)
