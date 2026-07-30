"""Optional neural layer (Neuropixels spikes) built on top of the behavioral core.

Imports :mod:`navigraph_analysis.behavior` / prep products but never the reverse.
Requires the ``[neural]`` extra for HMM modelling; the PSTH / PCA / place-field
functions here need only numpy/scipy.

- :mod:`~navigraph_analysis.neural.clock`       spike/video clock alignment
- :mod:`~navigraph_analysis.neural.psth`        event-aligned PSTHs
- :mod:`~navigraph_analysis.neural.population`  population rate matrix + PCA
- :mod:`~navigraph_analysis.neural.place_cells` graph-node place fields
"""

from .clock import build_trial_clock, gather_relative_spikes, frame_to_abs
from .psth import binned_and_smoothed_rate, psth_across_events
from .population import build_population_rate_matrix, run_pca
from .place_cells import node_rate_map, spatial_information

__all__ = [
    "build_trial_clock",
    "gather_relative_spikes",
    "frame_to_abs",
    "binned_and_smoothed_rate",
    "psth_across_events",
    "build_population_rate_matrix",
    "run_pca",
    "node_rate_map",
    "spatial_information",
]
