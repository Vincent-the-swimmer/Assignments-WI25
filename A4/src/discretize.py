import numpy as np


# Quantize the state space and action space
def quantize_state(state: dict, state_bins: dict) -> tuple:
    """
    
    Given the state and the bins for each state variable, quantize the state space.

    Args:
        state (dict): The state to be quantized.
        state_bins (dict): The bins used for quantizing each dimension of the state.

    Returns:
        tuple: The quantized representation of the state.
    """
    quant_state = tuple()
    for i in range(len(state['position'])):
        inds = np.digitize(state['position'][i], state_bins['position'][i])-1
        # tuple_part = (int(state_bins['position'][i][inds]), )
        tuple_part = (int(inds), )
        quant_state = quant_state + tuple_part
    for i in range(len(state['velocity'])):
        inds = np.digitize(state['velocity'][i], state_bins['velocity'][i])-1
        # tuple_part = (int(state_bins['velocity'][i][inds]), )
        tuple_part = (int(inds), )
        quant_state = quant_state + tuple_part
    # print(state["velocity"][0])
    return quant_state
    ...


def quantize_action(action: float, bins: list) -> int:
    """
    Quantize the action based on the provided bins. 
    """
    return np.digitize(action, bins)-1
    ...


