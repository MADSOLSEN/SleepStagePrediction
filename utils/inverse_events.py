from .binary_to_array import binary_to_array
from .semantic_formating import semantic_formating
import numpy as np

def inverse_events(output_size, events):
    events_ori = events
    if events.size > 0:
        if events.max() > 1:
            events = events / output_size
        event_sem = semantic_formating(output_size=output_size, events=events, sufficient_overlap=0)
        # inverse:
        event_sem_inverse = abs(event_sem - 1)
        inv_events = np.array(binary_to_array(event_sem_inverse))
    else:
        inv_events = np.array([])
    return inv_events
