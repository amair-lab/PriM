"""

    This could be a tool function case, the agent can have this method to call the prediction model.

"""

from typing import Dict, Union, Any
import requests
from requests.exceptions import RequestException
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def do_experiment(
        angle: float,
        curl: float,
        fiber_radius: float,
        height: float,
        helix_radius: float,
        n_turns: int,
        pitch: float,
        total_fiber_length: float,
        total_length: float
) -> Dict[str, Any]:
    """
    Performs a g-factor prediction experiment by sending parameters to a local prediction server.

    Args:
        angle (float): Angle in degrees
        curl (float): Curl value
        fiber_radius (float): Radius of the fiber
        height (float): Height value
        helix_radius (float): Radius of the helix
        n_turns (int): Number of turns
        pitch (float): Pitch value
        total_fiber_length (float): Total length of the fiber
        total_length (float): Total length value

    Returns:
        Dict[str, Any]: A dictionary containing:
            - status: 'success' or 'error'
            - predicted_g_factor: The predicted g-factor value (if successful)
            - message: Error message (if an error occurred)

    Raises:
        ValueError: If any parameters have invalid types or values
        ConnectionError: If the server is not accessible
        RequestException: For other request-related errors

    """
    # Validate float parameters
    float_params = {
        'angle': angle,
        'curl': curl,
        'fiber_radius': fiber_radius,
        'height': height,
        'helix_radius': helix_radius,
        'n_turns': n_turns,
        'pitch': pitch,
        'total_fiber_length': total_fiber_length,
        'total_length': total_length
    }

    for param_name, param_value in float_params.items():
        if not isinstance(param_value, (int, float)):
            raise ValueError(f"{param_name} must be a numeric value")

    # Construct the data dictionary for the request
    data = {
        'angle': float(angle),
        'curl': float(curl),
        'fiber_radius': float(fiber_radius),
        'height': float(height),
        'helix_radius': float(helix_radius),
        'n_turns': n_turns,
        'pitch': float(pitch),
        'total_fiber_length': float(total_fiber_length),
        'total_length': float(total_length)
    }

    try:
        # Make the request to the prediction server
        response = requests.post(
            'http://127.0.0.1:21500/predict',
            json=data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to prediction server at localhost:5000")
        return {
            'status': 'error',
            'message': 'Prediction server is not accessible. Please ensure it is running.'
        }

    except RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return {
            'status': 'error',
            'message': f'Request failed: {str(e)}'
        }

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }


if __name__ == '__main__':

    # Example usage
    result = do_experiment(
        angle=65.0,
        curl=0.4,
        fiber_radius=0.1,
        height=1.88,
        helix_radius=0.5,
        n_turns=2,
        pitch=0.74,
        total_fiber_length=10.0,
        total_length=8.0
    )

    if result['status'] == 'success':
        print(f"Predicted g-factor: {result['predicted_g_factor']}")
    else:
        print(f"Error: {result['message']}")