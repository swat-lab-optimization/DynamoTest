"""Utilities to orchestrate and summarize STL tracers.

`TracerMonitor` runs one or more tracers over streaming inputs (step-wise)
or over full episodes, tracking which tracers reported positive robustness and
providing lightweight bookkeeping across steps and episodes.
"""

from typing import List
from common.tracers import AbstractSignalTracer


class TracerMonitor:
    """Runs and aggregates multiple signal tracers.

    Attributes:
        tracers: List of concrete tracers to evaluate.
        tracer_dict: Per-step map of tracer names to boolean positivity flags.
        global_tracer_dict: Per-episode map of tracer names to positivity.
        positive_tracer_names: Names of tracers positive at the current step.
        global_positive_tracer_names: Names of tracers positive over the episode.
        _step: Current step index within the episode.
        episode: Episode counter incremented at each reset.
    """

    def __init__(self, tracer_list: List[AbstractSignalTracer]):
        """Initialize the monitor with a set of tracers.

        Args:
            tracer_list: Concrete `AbstractSignalTracer` instances to run.
        """
        self.tracers = tracer_list
        self._trace_index = 0  
        self._step = 0
        self.tracer_dict = {}  
        self.global_tracer_dict = {}  
        self.positive_flag = False
        self.positive_tracer_names = []
        self.global_positive_tracer_names = []
        self.all_trace = {} 
        self.episode = 0

    def monitor_step(self, input_trace):
        """Process a single step of inputs across all tracers.

        Evaluates each tracer on the latest signals and records which ones show
        non-negative robustness, while applying additional sanity checks for
        certain patterns (e.g., lane change heuristics for cut-in detection).

        Args:
            input_trace: Dict-like mapping of signal name to list of values,
                where the last element represents the current step.

        Returns:
            The robustness value returned by the last evaluated tracer after
            scaling. This is primarily for immediate feedback; aggregate state
            is stored in `tracer_dict` and `positive_tracer_names`.
        """
        self.positive_tracer_names = []
        self.tracer_dict[self._step] = {}
        self.positive_flag = False

        for tracer in self.tracers:
            # Initialize current step flag for this tracer
            self.tracer_dict[self._step][tracer.name] = False

            rob = tracer.evaluate_step(input_trace)
            # Only consider positive robustness and avoid first few unstable steps
            if rob >= 0 and self._step > 2:
                if "Cut" in tracer.name:
                    # Non-ego cut event: check adversary lane change magnitude
                    if (
                        "Ego" not in tracer.name
                        and abs(
                            input_trace["adv_lane"][-2] - input_trace["adv_lane"][-1]
                        )
                        > 1
                    ):
                        self.tracer_dict[self._step][tracer.name] = True
                        rob = rob / 10  # scale reward for step-wise use
                        if tracer.name not in self.positive_tracer_names:
                            self.positive_tracer_names.append(tracer.name)
                    # Ego cut event: check ego lane change magnitude
                    elif (
                        "Ego" in tracer.name
                        and abs(
                            input_trace["ego_lane"][-2] - input_trace["ego_lane"][-1]
                        )
                        > 1
                    ):
                        self.tracer_dict[self._step][tracer.name] = True
                        rob = rob / 10
                        if tracer.name not in self.positive_tracer_names:
                            self.positive_tracer_names.append(tracer.name)
                elif "FrontSlowDownSameLaneTracer" in tracer.name:
                    # Either the vehicle ahead slowed down, or ego is faster
                    # after a cut-in event was detected recently
                    if (
                        input_trace["adv_speed"][-2] > input_trace["adv_speed"][-1]
                    ) or (
                        (
                            ("CutInTracer" in self.positive_tracer_names)
                            or self.tracer_dict[self._step - 1]["CutInTracer"]
                        )
                        and input_trace["ego_speed"][-1] > input_trace["adv_speed"][-1]
                    ):
                        self.tracer_dict[self._step][tracer.name] = True
                        rob = rob / 10
                        if tracer.name not in self.positive_tracer_names:
                            self.positive_tracer_names.append(tracer.name)
                        print(
                            f"Positive tracer names (local): {self.positive_tracer_names}"
                        )
                else:
                    # Default path: positive monitor result marks this step
                    self.tracer_dict[self._step][tracer.name] = True
                    rob = rob / 10
                    if tracer.name not in self.positive_tracer_names:
                        self.positive_tracer_names.append(tracer.name)
                    print(
                        f"Positive tracer names (local): {self.positive_tracer_names}"
                    )

        # Advance to next step for subsequent calls
        self._step += 1
        return rob

    def monitor_episode(self, input_trace):
        """Evaluate all tracers over a full episode trace.

        Args:
            input_trace: Mapping of signal names to full episode sequences.

        Returns:
            The last robustness value computed among the tracers. Episode-wide
            positives are recorded in `global_tracer_dict` and
            `global_positive_tracer_names`.
        """
        self.global_step = 0  # reserved for future use
        self.positive_tracer_names = []
        for tracer in self.tracers:
            self.global_tracer_dict[tracer.name] = False
            rob = -1
            rob = tracer.evaluate(input_trace)
            if rob >= 0:
                print(f"Tracer {tracer.name} evaluated with reward: {rob}")
                self.global_tracer_dict[tracer.name] = True
                self.global_positive_tracer_names.append(tracer.name)

        return rob

    def reset(self):
        """Reset per-step and per-episode bookkeeping and tracers themselves."""
        self._trace_index = 0
        self._step = 0
        self.tracer_dict = {}
        self.global_tracer_dict = {}
        self.positive_flag = False
        self.positive_tracer_names = []
        self.global_positive_tracer_names = []
        self.all_trace = {}
        for tracer in self.tracers:
            tracer.reset()
        self.episode += 1

    def save(self, filepath):
        """Persist the per-step positivity map to a JSON file.

        Args:
            filepath: Destination path for the JSON output.
        """
        import json

        with open(filepath, "w") as f:
            json.dump(self.tracer_dict, f, indent=4)
        print(f"Tracer monitor saved to {filepath}")
