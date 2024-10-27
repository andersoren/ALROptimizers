import pytest
import torch
import torch.nn as nn
from Custom_Optimizers import SRPROP


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def setup_optimizer():
    model = SimpleModel()
    optimizer = SRPROP(model.parameters())
    return model, optimizer


def test_optimizer_initialization():
    """Test that optimizer initializes correctly with valid parameters."""
    model = SimpleModel()
    optimizer = SRPROP(model.parameters(), lr=0.01)
    assert optimizer.defaults['lr'] == 0.01
    assert optimizer.defaults['weight_decay'] == 0  # default value
    assert len(optimizer.param_groups) == 1


def test_invalid_parameters():
    """Test that optimizer raises appropriate errors for invalid parameters."""
    model = SimpleModel()
    with pytest.raises(ValueError):
        SRPROP(model.parameters(), lr=-0.01)
    with pytest.raises(ValueError):
        SRPROP(model.parameters(), L=5, M=2)
    with pytest.raises(ValueError):
        SRPROP(model.parameters(), etas=(2, 0.2))


def test_optimization_step(setup_optimizer):
    """Test that optimization step changes parameters and reduces loss."""
    model, optimizer = setup_optimizer
    
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    criterion = nn.MSELoss()
    
    initial_params = [param.clone() for param in model.parameters()]
    
    def closure():
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        return loss
    
    initial_loss = closure()
    for _ in range(5):
        optimizer.step(closure)
    
    final_loss = closure()
    
    for p, initial_p in zip(model.parameters(), initial_params):
        assert not torch.allclose(p, initial_p)
    
    assert final_loss < initial_loss

def test_state_dict(setup_optimizer):
    """Test that optimizer state can be properly saved and loaded."""
    model, optimizer = setup_optimizer
    
    X = torch.randn(10, 10)
    y = torch.randn(10, 1)
    criterion = nn.MSELoss()
    
    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    
    state_dict = optimizer.state_dict()
    
    new_optimizer = SRPROP(model.parameters())
    new_optimizer.load_state_dict(state_dict)
    
    assert optimizer.state_dict()['param_groups'] == new_optimizer.state_dict()['param_groups']
    
    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()