import pytest
import torch
from torch import nn

from hornets import HorNetsArchitecture


@pytest.fixture
def model():
    return HorNetsArchitecture(
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


@pytest.fixture
def model_relu():
    return HorNetsArchitecture(
        dim=10,
        outpt=3,
        num_rules=5,
        num_features=10,
        exp_param=2,
        feature_names=[f"feature_{i}" for i in range(10)],
        activation="relu",
        order=2,
        device=torch.device("cpu"),
    )


class TestInitialization:
    def test_basic_attributes(self, model):
        assert model.num_features == 10
        assert model.num_tars == 3
        assert len(model.comb_indices) == 5
        assert model.activation == "polyclip"
        assert model.device.type == "cpu"

    def test_comb_space_shape(self, model):
        assert model.comb_space.shape == (2, 5)

    def test_comb_scores_initialized_to_zero(self, model):
        assert torch.all(model.comb_scores == 0)

    def test_num_rules_exceeds_combinations(self):
        m = HorNetsArchitecture(
            dim=4, outpt=2, num_rules=100, num_features=4,
            exp_param=1, order=2, device=torch.device("cpu"),
        )
        # C(4,2) = 6, so only 6 rules even though 100 requested
        assert len(m.comb_indices) == 6

    def test_order_equals_num_features(self):
        m = HorNetsArchitecture(
            dim=3, outpt=2, num_rules=10, num_features=3,
            exp_param=1, order=3, device=torch.device("cpu"),
        )
        # C(3,3) = 1
        assert len(m.comb_indices) == 1


class TestPolyClip:
    def test_output_range(self, model):
        x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        out = model.polyClip(x)
        assert torch.all(out >= -1) and torch.all(out <= 1)

    def test_hard_rounding(self, model):
        x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        out = model.polyClip(x, hard=True)
        assert all(val in [-1, 0, 1] for val in out.tolist())

    def test_zero_input(self, model):
        x = torch.zeros(5)
        out = model.polyClip(x)
        assert torch.all(out == 0)

    def test_preserves_gradient(self, model):
        x = torch.tensor([0.5], requires_grad=True)
        out = model.polyClip(x)
        out.backward()
        assert x.grad is not None


class TestGetRoute:
    def test_binary_input(self, model):
        x = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.float32)
        assert model.get_route(x) == 1

    def test_continuous_input(self, model):
        x = torch.tensor([[0.5, 0.7], [1.0, 2.0]], dtype=torch.float32)
        assert model.get_route(x) == 0

    def test_all_zeros(self, model):
        x = torch.zeros(3, 4)
        assert model.get_route(x) == 0

    def test_all_ones(self, model):
        x = torch.ones(3, 4)
        assert model.get_route(x) == 0

    def test_mixed_binary_and_float(self, model):
        x = torch.tensor([[0, 1, 0.5]], dtype=torch.float32)
        assert model.get_route(x) == 0


class TestForward:
    def test_binary_route_shape(self, model):
        x = torch.randint(0, 2, (4, 10)).float()
        out = model.forward(x)
        assert out.shape == (4, 3)

    def test_continuous_route_shape(self, model):
        x = torch.randn(4, 10)
        out = model.forward(x)
        assert out.shape == (4, 3)

    def test_binary_output_is_log_softmax(self, model):
        x = torch.randint(0, 2, (4, 10)).float()
        out = model.forward(x)
        # log_softmax values should be <= 0
        assert torch.all(out <= 0)
        # exp should sum to ~1
        probs = torch.exp(out)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_with_num_samples(self, model):
        x = torch.randint(0, 2, (4, 10)).float()
        out = model.forward(x, num_samples=2)
        assert out.shape == (4, 3)

    def test_relu_activation(self, model_relu):
        x = torch.randint(0, 2, (4, 10)).float()
        out = model_relu.forward(x)
        assert out.shape == (4, 3)

    def test_comb_scores_accumulate(self, model):
        model.reset_comb_scores()
        x = torch.randint(0, 2, (4, 10)).float()
        model.forward(x)
        scores_after_one = model.comb_scores.clone()
        model.forward(x)
        scores_after_two = model.comb_scores.clone()
        assert not torch.all(scores_after_one == scores_after_two)

    def test_reset_comb_scores(self, model):
        x = torch.randint(0, 2, (4, 10)).float()
        model.forward(x)
        model.reset_comb_scores()
        assert torch.all(model.comb_scores == 0)


class TestEmbed:
    def test_binary_embed_shape(self, model):
        x = torch.randint(0, 2, (4, 10)).float()
        emb = model.embed(x)
        assert emb.shape == (4, len(model.comb_indices))

    def test_continuous_embed_shape(self, model):
        x = torch.randn(4, 10)
        emb = model.embed(x)
        assert emb.shape == (4, 10)

    def test_binary_embed_is_softmax(self, model):
        model.eval()
        x = torch.randint(0, 2, (4, 10)).float()
        emb = model.embed(x)
        sums = emb.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_embed_matches_forward_route(self, model):
        model.eval()
        x = torch.randint(0, 2, (4, 10)).float()
        emb = model.embed(x)
        out = model.forward(x)
        # Forward applies out_linear + log_softmax on top of embed
        assert emb.shape[0] == out.shape[0]


class TestBackprop:
    def test_gradients_exist(self, model):
        x = torch.randint(0, 2, (4, 10)).float()
        y = torch.randint(0, 3, (4,))
        out = model.forward(x)
        loss = nn.NLLLoss()(out, y)
        loss.backward()
        assert model.comb_space.grad is not None
        assert not torch.all(model.comb_space.grad == 0)

    def test_loss_decreases(self, model):
        x = torch.randint(0, 2, (8, 10)).float()
        y = torch.randint(0, 3, (8,))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.NLLLoss()

        model.train()
        initial_loss = None
        for _ in range(20):
            out = model.forward(x)
            loss = loss_fn(out, y)
            if initial_loss is None:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert loss.item() < initial_loss


class TestGetRules:
    def test_returns_list(self, model):
        x = torch.randint(0, 2, (4, 10)).float()
        model.forward(x)
        rules = model.get_rules()
        assert isinstance(rules, list)
        assert len(rules) == 3

    def test_rule_structure(self, model):
        x = torch.randint(0, 2, (4, 10)).float()
        model.forward(x)
        rules = model.get_rules(top_k=2)
        assert len(rules) == 2
        for rule in rules:
            assert "features" in rule
            assert "score" in rule
            assert isinstance(rule["features"], list)
            assert isinstance(rule["score"], float)
