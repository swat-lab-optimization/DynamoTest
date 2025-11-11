"""Signal Temporal Logic (STL) tracers for vehicle interactions.

This module provides a base tracer and several concrete tracers that configure
STL formulas using the `rtamt` library to monitor relationships between an ego
vehicle and an adversary vehicle over time. Implementations expose utilities to
accumulate signal traces from simulation state and to evaluate robustness
metrics stepwise or over entire traces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import rtamt
from highway_env.vehicle.behavior import ControlledVehicle
import numpy as np


class AbstractSignalTracer(ABC):
    """Base class for STL-based signal tracers.

    The tracer encapsulates an STL specification, maintains input traces for
    relevant signals, and exposes evaluation helpers. Subclasses define the
    concrete specification via `_define_spec`.

    Attributes:
        sampling_period: Discrete sampling period used by the specification.
        spec: The underlying `rtamt.StlDiscreteTimeSpecification` instance.
        input_trace: Accumulated signal values keyed by signal name.
        current_time_step: Number of samples appended to `input_trace`.
        time: Internal discrete time index used for stepwise evaluation.
    """

    def __init__(self, sampling_period: float = 0.1):
        """Initialize the tracer with a sampling period.

        Args:
            sampling_period: Discrete sampling period for the STL monitor in
                seconds.
        """
        self.sampling_period = sampling_period
        self.spec = None
        self.input_trace = {}
        self.current_time_step = 0
        self._init_spec()
        self.time = 0

    @abstractmethod
    def _define_spec(self):
        """Populate the STL formula for this tracer.

        Subclasses must assign the STL formula string to `self.spec.spec` and
        may tune parameters used by the specification.
        """
        pass

    def _init_spec(self):
        """Create and configure the STL specification instance.

        Declares input/output variables, sets the sampling period, and assigns
        default metadata required by the `rtamt` monitor. Subclasses will later
        define the concrete formula via `_define_spec`.
        """
        self.spec = rtamt.StlDiscreteTimeSpecification(
            semantics=rtamt.Semantics.STANDARD
        )
        self.spec.name = "Cut-in Detection"

        self.spec.declare_var("adv_x", "float")
        self.spec.declare_var("adv_lane", "float")
        self.spec.declare_var("ego_x", "float")
        self.spec.declare_var("ego_lane", "float")
        self.spec.declare_var("ego_speed", "float")
        self.spec.declare_var("adv_speed", "float")
        self.spec.declare_var("rob", "float")
        self.spec.declare_var("ego_lane_id", "int")
        self.spec.declare_var("adv_lane_id", "int")
        self.spec.declare_var("adv_heading", "float")

        self.spec.set_var_io_type("adv_x", "input")
        self.spec.set_var_io_type("adv_lane", "input")
        self.spec.set_var_io_type("ego_x", "input")
        self.spec.set_var_io_type("ego_lane", "input")
        self.spec.set_var_io_type("ego_speed", "input")
        self.spec.set_var_io_type("adv_speed", "input")
        self.spec.set_var_io_type("ego_lane_id", "input")
        self.spec.set_var_io_type("adv_lane_id", "input")
        self.spec.set_var_io_type("adv_heading", "input")

        self.spec.set_var_io_type("rob", "output")

        self.spec.set_sampling_period(self.sampling_period)

    def reset(self):
        """Clear accumulated inputs and rewind internal time indices."""
        self.current_time_step = 0
        self.input_trace = {}
        self.time = 0

        # self.spec = None

    def _parse_spec(self):
        """Parse the configured STL specification.

        Parses the current `self.spec` and prepares it for evaluation.

        Raises:
            ValueError: If the specification has not been initialized.
        """
        if self.spec is None:
            raise ValueError("STL specification is not defined.")
        self.spec.parse()

    def evaluate2(
        self, input_trace: Dict[str, List[Tuple[float, float]]]
    ) -> List[Tuple[float, float]]:
        """Evaluate the specification over an entire trace at once.

        Args:
            input_trace: Mapping from signal name to a list of `(time, value)`
                pairs for the whole trace.

        Returns:
            A list of `(time, robustness)` pairs as returned by `rtamt`.

        Raises:
            ValueError: If the specification is not initialized.
        """
        if self.spec is None:
            raise ValueError("Specification not initialized.")
        trace_list = [[k, v] for k, v in input_trace.items()]

        return self.spec.evaluate(input_trace)

    def evaluate(
        self, input_trace: Dict[str, List[Tuple[float, float]]]
    ) -> float:
        """Evaluate the specification by iterating over a provided trace.

        This helper feeds each time step to the monitor and returns the maximum
        robustness observed.

        Args:
            input_trace: Mapping from signal name to sampled values for the
                entire discrete-time horizon.

        Returns:
            float: Maximum robustness value across all evaluated steps.

        Raises:
            ValueError: If the specification is not initialized.
        """
        if self.spec is None:
            raise ValueError("Specification not initialized.")
        trace_list = [(k, v) for k, v in input_trace.items()]
        rob_list = []
        for i in range(len(input_trace["adv_x"])):
            rob = self.spec.update(
                i,
                [
                    ("adv_x", input_trace["adv_x"][i]),
                    ("adv_lane", input_trace["adv_lane"][i]),
                    ("ego_x", input_trace["ego_x"][i]),
                    ("ego_lane", input_trace["ego_lane"][i]),
                ],
            )
            if rob >= 0:
                print(
                    f"Positive robustness value: {rob} at time step {i} for {self.name}"
                )
            rob_list.append(rob)
        rob = max(rob_list)

        return rob

    def evaluate_step(
        self, input_trace: Dict[str, List[float]]
    ) -> float:
        """Evaluate the specification for the latest time step.

        Feeds the most recent sample from `input_trace` to the monitor and
        returns the resulting robustness value.

        Args:
            input_trace: Mapping from signal name to lists of sampled scalar
                values. The last element of each list is treated as the latest
                time step.

        Returns:
            float: Robustness value computed for the current step.
        """
        try:
            rob = self.spec.update(
                self.time,
                [
                    ("adv_x", input_trace["adv_x"][self.time]),
                    ("adv_lane", input_trace["adv_lane"][self.time]),
                    ("ego_x", input_trace["ego_x"][self.time]),
                    ("ego_lane", input_trace["ego_lane"][self.time]),
                ],
            )
        except Exception as e:
            print(f"Error updating specification: {e}")
            rob = -1
        rob = np.clip(rob, -100, 100)  # Limit robustness value to a maximum of 20
        if rob >= 0:
            print(
                f"Positive robustness value update: {rob} at time step {self.time} for {self.name}"
            )
            rob *= 2
            if rob == 0:
                rob = 2
        elif rob == 100:
            rob = -100
        # https://github.com/ezapridou/carla-acc/blob/master/acc_runtime_monitoring.py
        self.time += 1
        if rob < 0:
            rob /= 1000
        return rob

    def update(self, ego_veh: ControlledVehicle, adv_veh: ControlledVehicle):
        """Append a new sample to the internal input trace.

        Reads the current state from the provided vehicles and appends values
        for all declared signals. Lateral positions are clipped to valid lane
        bounds before rounding.

        Args:
            ego_veh: The ego vehicle to be monitored.
            adv_veh: The other vehicle interacting with the ego vehicle.
        """
        adv_veh.position[1] = np.clip(adv_veh.position[1], 0, 4)
        ego_veh.position[1] = np.clip(ego_veh.position[1], 0, 4)
        if self.current_time_step == 0:
            self.input_trace["adv_x"] = [float(adv_veh.position[0])]
            self.input_trace["adv_lane"] = [(round(adv_veh.position[1]))]
            self.input_trace["ego_x"] = [float(ego_veh.position[0])]
            self.input_trace["ego_lane"] = [(round(ego_veh.position[1]))]
            self.input_trace["ego_speed"] = [(ego_veh.speed)]
            self.input_trace["adv_speed"] = [(adv_veh.speed)]
            self.input_trace["ego_lane_id"] = [int(ego_veh.lane_index[2])]
            self.input_trace["adv_lane_id"] = [int(adv_veh.lane_index[2])]
            self.input_trace["adv_heading"] = [float(adv_veh.heading)]
        else:
            self.input_trace["adv_x"].append(float(adv_veh.position[0]))
            self.input_trace["adv_lane"].append((round(adv_veh.position[1])))
            self.input_trace["ego_x"].append(float(ego_veh.position[0]))
            self.input_trace["ego_lane"].append((round(ego_veh.position[1])))
            self.input_trace["ego_speed"].append((ego_veh.speed))
            self.input_trace["adv_speed"].append((adv_veh.speed))
            self.input_trace["ego_lane_id"].append(int(ego_veh.lane_index[2]))
            self.input_trace["adv_lane_id"].append(int(adv_veh.lane_index[2]))
            self.input_trace["adv_heading"].append(float(adv_veh.heading))


        self.current_time_step += 1


class SameSpeedEgoTracer(AbstractSignalTracer):
    """Tracer configured to detect near-constant ego speed behavior."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 30.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "SameSpeedEgoTracer"

    def _define_spec(self):
        """Assign the STL formula that captures stable ego speed."""
        self.spec.spec = """
           rob = always[0:10](  (prev ego_speed - ego_speed < 0.2) and (prev ego_speed - ego_speed > 0 ) )
        """  # and adv_x > ego_x


class CutInTracer(AbstractSignalTracer):
    """Tracer for detecting when another vehicle merges into the ego lane ahead."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 30.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        self.lane_id_tol = 0.5
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "CutInTracer"

    def _define_spec(self):
        """Assign the STL formula that encodes a cut-in ahead of ego."""
        self.spec.spec = f"""
            rob = (  
           (prev(abs(adv_lane - ego_lane))  > {self.lane_tol}) and 
           (abs(adv_lane - ego_lane) <= {self.lane_tol} )
            and ((adv_x - ego_x) < {self.d_safe} )
            and (adv_x > ego_x + 5)  )
            """  # and adv_x > ego_x


class EgoCutInTracer(AbstractSignalTracer):
    """Tracer for detecting when the ego vehicle merges into another lane ahead."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 30.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "EgoCutInTracer"

    def _define_spec(self):
        """Assign the STL formula that encodes an ego cut-in event."""
        self.spec.spec = f"""
           rob = (  
           (prev(abs(adv_lane - ego_lane))  > {self.lane_tol}) and 
           (abs(adv_lane - ego_lane) < {self.lane_tol} )
           and ((adv_x - ego_x) < {self.d_safe} )
           and (adv_x > ego_x)  )
        """  # and adv_x > ego_x


class CutInSideTracer(AbstractSignalTracer):
    """Tracer for detecting merges resulting in side-by-side positioning."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 30.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "CutInSideTracer"

    def _define_spec(self):
        """Assign the STL formula that captures a side-by-side cut-in."""
        self.spec.spec = f"""
           rob = (  
           (prev(abs(adv_lane - ego_lane))  > {self.lane_tol}) and 
           (abs(adv_lane - ego_lane) < {self.lane_tol} )
           and ((adv_x - ego_x) < {self.d_safe} )
           and (adv_x < ego_x+5) and (adv_x - ego_x > -10) )
        """  # and adv_x > ego_x


class EgoCutInSideTracer(AbstractSignalTracer):
    """Tracer for detecting ego merges that end side-by-side with another vehicle."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 30.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "EgoCutInSideTracer"

    def _define_spec(self):
        """Assign the STL formula for ego side-by-side cut-in events."""
        self.spec.spec = f"""
           rob = (  
           (prev(abs(adv_lane - ego_lane))  > {self.lane_tol}) and 
           (abs(adv_lane - ego_lane) < {self.lane_tol} )
           and ((adv_x - ego_x) < {self.d_safe} )
           and (adv_x < ego_x) and (ego_x - adv_x < 10)  )
        """  # and adv_x > ego_x


class CutOutTracer(AbstractSignalTracer):
    """Tracer for detecting lane departures from the ego lane ahead."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 30.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "CutOutTracer"

    def _define_spec(self):
        """Assign the STL formula that captures cut-out events ahead of ego."""
        self.spec.spec = f"""
            rob = (  ((prev(abs(adv_lane - ego_lane))  < {self.lane_tol}) and 
            (abs(adv_lane - ego_lane) > {self.lane_tol})  ) 
            and ((adv_x - ego_x) < {self.d_safe}) 
            and ((adv_x - ego_x) > 2)   ) 
        """


class EgoCutOutTracer(AbstractSignalTracer):
    """Tracer for detecting when the ego vehicle leaves its lane near another vehicle."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 30.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "EgoCutOutTracer"

    def _define_spec(self):
        """Assign the STL formula that captures ego lane departures near others."""
        self.spec.spec = f"""
            rob = (  ((prev(abs(adv_lane - ego_lane))  < {self.lane_tol}) and 
            (abs(adv_lane - ego_lane) > {self.lane_tol})  ) 
            and ((adv_x - ego_x) < {self.d_safe}) 
            and ((adv_x - ego_x) > 2)   ) 
        """


class BehindSpeedUpTracer(AbstractSignalTracer):
    """Tracer for detecting closing gaps behind the ego vehicle."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 20.0):
        self.d_safe = d_safe
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "BehindSpeedUpTracer"

    def _define_spec(self):
        """Assign the STL formula that captures a vehicle speeding up behind ego."""
        self.spec.spec = f"""
            rob = ( eventually[0:40] ( always[0:3]( 
   
            abs(adv_x - ego_x) < {self.d_safe} 
            and ( ego_x - adv_x) > 2  and prev(adv_x) > adv_x ) )
            )
        """


class FrontSlowDownSameLaneTracer(AbstractSignalTracer):
    """Tracer for detecting slowdowns ahead in the same lane."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 40.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "FrontSlowDownSameLaneTracer"

    def _define_spec(self):
        """Assign the STL formula that encodes slowdowns ahead in the same lane."""
        self.spec.spec = f"""
            rob = (  ((
            (abs(adv_lane - ego_lane) < {self.lane_tol} ) 
            and (abs(adv_x - ego_x) < {self.d_safe} )
            and ((adv_x - ego_x) > 2) and  (prev adv_speed > adv_speed ) ))
            )
        """
        # and (prev(adv_x) < adv_x)


class FrontSlowDownDifferentLaneTracer(AbstractSignalTracer):
    """Tracer for detecting slowdowns ahead in a different lane."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 40.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "FrontSlowDownDifferentLaneTracer"

    def _define_spec(self):
        """Assign the STL formula that encodes slowdowns ahead in other lanes."""
        self.spec.spec = f"""
            rob = (  ( ( 
            (abs(adv_lane - ego_lane) > {self.lane_tol} ) 
            and (abs(adv_x - ego_x) < {self.d_safe} )
            and ((adv_x - ego_x) > 2)  and (prev adv_speed > adv_speed) ) )
            )
        """


class BehindSameLaneTracer(AbstractSignalTracer):
    """Tracer for detecting vehicles following closely in the same lane."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 15.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "BehindSameLaneTracer"

    def _define_spec(self):
        """Assign the STL formula that encodes a close follower in the same lane."""
        self.spec.spec = f"""
            rob = ( eventually[0:40] ( always[0:5]( 
            (abs(adv_lane - ego_lane) < {self.lane_tol} ) 
            and abs(adv_x - ego_x) < {self.d_safe} 
            and adv_x < ego_x ) ) )
        """
        #


class BehindDifferentLaneTracer(AbstractSignalTracer):
    """Tracer for detecting vehicles following closely in a different lane."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 15.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "BehindDifferentLaneTracer"

    def _define_spec(self):
        """Assign the STL formula that encodes a close follower in another lane."""
        self.spec.spec = f"""
            rob = ( eventually[0:40] ( always[0:5]( 
            (abs(adv_lane - ego_lane) > {self.lane_tol} ) 
            and abs(adv_x - ego_x) < {self.d_safe} 
            and adv_x < ego_x ) ) )
        """


class FrontSameLaneTracer(AbstractSignalTracer):
    """Tracer for detecting a vehicle ahead within a distance in the same lane."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 30.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "FrontSameLaneTracer"

    def _define_spec(self):
        """Assign the STL formula that encodes a lead vehicle in the same lane."""
        self.spec.spec = f"""
            rob = ( eventually[0:200] ( always[0:1]( 
            (abs(adv_lane - ego_lane) < {self.lane_tol} ) 
            and abs(adv_x - ego_x) < {self.d_safe} 
            and adv_x - ego_x > 2 ) ) )
        """


class FrontDifferentLaneTracer(AbstractSignalTracer):
    """Tracer for detecting a vehicle ahead within a distance in another lane."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 30.0):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "FrontDifferentLaneTracer"

    def _define_spec(self):
        """Assign the STL formula that encodes a lead vehicle in another lane."""
        self.spec.spec = f"""
            rob = ( eventually[0:40] ( always[0:1]( 
            (abs(adv_lane - ego_lane) > {self.lane_tol} ) 
            and abs(adv_x - ego_x) < {self.d_safe} 
            and (adv_x - ego_x > 2)      ) ) ) 
        """
        # and prev(adv_x) == adv_x


class SideTracer(AbstractSignalTracer):
    """Tracer for detecting side-by-side proximity in adjacent lanes."""
    def __init__(self, sampling_period: float = 0.2, d_safe: float = 5):
        self.d_safe = d_safe
        self.lane_tol = 1
        super().__init__(sampling_period)
        self._define_spec()
        self._parse_spec()
        self.spec.pastify()
        self.name = "SideTracer"

    def _define_spec(self):
        """Assign the STL formula that encodes lateral proximity in neighboring lanes."""
        self.spec.spec = f"""
            rob = ( eventually[0:40] ( always[0:5]( 
            (abs(adv_lane - ego_lane) > {self.lane_tol} ) 
            and abs(adv_x - ego_x) < {self.d_safe} 
           ) ) )
        """
