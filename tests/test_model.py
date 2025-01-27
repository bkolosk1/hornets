import pytest
import torch
from torch import nn
from hornets import HorNetsArchitecture


@pytest.fixture
def hornet_default():
    model = HorNetsArchitecture(
        dim=10,
        outpt=3,
        num_rules=5,
        num_features=10,
        exp_param=2,
        feature_names=[f"feature_{i}" for i in range(10)],
        activation="polyclip",
        order=2,
        device=torch.device("cpu"),
    )
    return model


def test_hornet_initialization(hornet_default):
    model = hornet_default
    assert model.num_features == 10
    assert model.num_tars == 3
    assert len(model.comb_indices) == 5
    assert model.comb_space.shape == (model.comb_space.shape[0], 5)
    assert model.activation == "polyclip"
    assert model.device.type == "cpu"


def test_hornet_forward_binary_route(hornet_default):
    model = hornet_default
    x_bin = torch.randint(0, 2, (4, 10)).float()
    output_bin = model.forward(x_bin)
    assert output_bin.shape == (4, 3)


def test_polyclip(hornet_default):
    model = hornet_default
    x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
    x_polyclip = model.polyClip(x)
    assert torch.all(x_polyclip >= -1) and torch.all(x_polyclip <= 1)
    x_rounded = model.polyClip(x, hard=True)
    assert all(val in [-1, 0, 1] for val in x_rounded.tolist())


def test_get_route(hornet_default):
    model = hornet_default
    x_bin = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.float32)
    route_bin = model.get_route(x_bin)
    assert route_bin == 1
    x_cont = torch.tensor([[0.5, 0.7, 0.2], [1.0, 2.0, 3.0]],
                          dtype=torch.float32)
    route_cont = model.get_route(x_cont)
    assert route_cont == 0


def test_get_rules(hornet_default, capsys):
    model = hornet_default
    x_bin = torch.randint(0, 2, (4, 10)).float()
    _ = model.forward(x_bin)
    model.get_rules()
    captured = capsys.readouterr()
    assert "Feature comb:" in captured.out
    assert "score:" in captured.out


def test_model_backprop(hornet_default):
    model = hornet_default
    x_bin = torch.randint(0, 2, (4, 10)).float()
    y_true = torch.randint(0, 3, (4, ))
    out = model.forward(x_bin)
    loss_fn = nn.NLLLoss()
    loss = loss_fn(out, y_true)
    loss.backward()
    assert True
