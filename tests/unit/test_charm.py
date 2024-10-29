from unittest.mock import Mock
from unittest.mock import patch
import pytest
from src.charm import MountainCarCharm

from ops.testing import Harness
from ops.charm import ActionEvent

@pytest.fixture
def harness():
    # Set up the testing harness for the charm
    harness = Harness(MountainCarCharm)
    harness.begin()
    yield harness
    harness.cleanup()

class MockSecret:
    def get_content(self):
        return {}

def test_my_custom_action_success(harness):
    # Create a mock event for the action
    action_event = Mock(spec=ActionEvent)
    action_event.params = {
        'feature-type': 'NRB',
        'alpha-list': ["0.025"],
        'gamma-list': ["0.95"],
        'epsilon-list': ["0.5"],
        'polynomial-dimension-list': ["1"],
        'protos-per-dimension-list': ["8"],
        'training-sessions': 1,
        'simulations': 1
    }
    # I have to mock the secret from database
    secret = Mock(spec_set=MockSecret)
    harness.charm.model.get_secret = Mock(return_value=secret)

    # Mocking the persist_trajectory_sizes_mean
    with patch("ml.mdp_semi_gradient_mountain_car_td_0.persist_trajectory_sizes_mean") as mock_persist:
        # Call the _on_simulation_action method directly with the mock event
        harness.charm._on_simulation_action(action_event)
        # Assert calls or behavior on the mock
        mock_persist.assert_called_once()

    # Get the arguments with which set_results was called
    set_results_args = action_event.set_results.call_args[0][0]

    assert set_results_args.get('success')

