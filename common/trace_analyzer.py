class TraceAnalyzer:
    def __init__(self, max_duration=40):
        # self.trace_dict = trace_dict
        self.max_duration = max_duration
        self.all_stats = {
            "EgoCutInTracer": [],
            "EgoCutInSideTracer": [],
            "EgoCutOutTracer": [],
            "CutInTracer": [],
            "CutInSideTracer": [],
            "FrontSlowDownSameLaneTracer": [],
        }

    def get_safe_distance(self, v_lead: float, v_follow: float) -> float:
        """
        Calculate a safe distance based on the speed of the vehicle.
        The formula is a simplified version of the 2-second rule.
        """

        deceleration = 5
        min_distance = 5
        reaction_time = 0.2  # sec for autonomous vehicle
        safe_distance = (
            (v_lead**2 - v_follow**2) / (2 * deceleration)
            + v_follow * reaction_time
            + min_distance
        )
        return safe_distance

    def time_to_collision(
        self, v_lead: float, v_follow: float, distance: float
    ) -> float:
        """
        Calculate the time to collision based on the speed of the lead and follow vehicles and the distance between them.
         (xF – xL - lF) / (vF - vL)
        """
        relative_speed = v_lead - v_follow
        return (distance - 5) / relative_speed

    def analyze(self, trace_dict: dict, all_frames_dict: dict, crashed=True) -> None:
        # Perform analysis on the trace_dict
        event_step = None
        found = False
        tracer_found = None
        ego_fault = False
        info = None
        metadata = {"adv_ego_dist": None, "safe_distance": None, "ttc": None}
        for step in reversed(trace_dict):
            for tracer in trace_dict[step]:
                if (
                    tracer in ["CutInTracer", "FrontSlowDownSameLaneTracer"]
                    and not trace_dict[step][tracer]
                ):
                    try:
                        if trace_dict[step + 1][tracer]:
                            event_frame = all_frames_dict[step + 1]
                            if event_frame["ego_speed"] > event_frame["adv_speed"]:
                                event_step = step + 1
                            else:
                                event_step = step + 2

                            found = True
                            tracer_found = tracer
                            break
                    except Exception as e:
                        continue

            if found:
                break

        if event_step is not None:
            event_step = min(event_step, len(all_frames_dict) - 1)
            event_frame = all_frames_dict[event_step]
            event_frame["crashed"] = crashed
            if event_frame["ego_speed"] > event_frame["adv_speed"]:
                safe_distance = self.get_safe_distance(
                    event_frame["ego_speed"], event_frame["adv_speed"]
                )
                ttc = self.time_to_collision(
                    event_frame["ego_speed"],
                    event_frame["adv_speed"],
                    event_frame["adv_ego_distance"],
                )
                observed_distance = event_frame["adv_ego_distance"]
                print(
                    f"Event detected {tracer_found} at step {event_step} with safe distance {safe_distance} and observed distance {observed_distance}"
                )
                ego_cut_in = trace_dict[event_step]["EgoCutInTracer"]
                event_frame["safe_distance"] = safe_distance
                event_frame["ttc"] = ttc
                print(f"Time to collision: {ttc}")
                metadata["adv_ego_dist"] = event_frame["adv_ego_distance"]
                metadata["safe_distance"] = safe_distance
                metadata["ttc"] = ttc

                if (
                    ego_cut_in and observed_distance < safe_distance
                ):  # TODO: consider a case where ego cut in and adversry cut in happen at the same time
                    print("Ego unsafe cut in detected")
                    ego_fault = True
                    info = "EgoCutInTracer"

                    self.all_stats["EgoCutInTracer"].append(event_frame)
                else:
                    self.all_stats[tracer_found].append(event_frame)
                    if observed_distance > safe_distance:
                        print("Adversary cut in / slowed down at a safe distance")
                        ego_fault = True
                        info = tracer_found
                    else:
                        info = tracer_found
                        print("Adversary cut in / slowed down at an unsafe distance")

        ego_side_cut_in = [
            trace_dict[step]["EgoCutInSideTracer"] for step in trace_dict
        ]
        adv_side_cut_in = [trace_dict[step]["CutInSideTracer"] for step in trace_dict]
        ego_cut_out = [trace_dict[step]["EgoCutOutTracer"] for step in trace_dict]
        last_frame = list(all_frames_dict.values())[-1]
        adv_crash_behind = (
            last_frame["adv_ego_distance"] < -4.9
            and abs(last_frame["adv_heading"]) < 0.1
        )

        if adv_crash_behind:
            ego_fault = False
            info = "AdvHitFromBehind"
            print("Adversary hit from behind detected")

            event_frame = last_frame
            event_frame["crashed"] = crashed

        elif any(adv_side_cut_in):
            print("Adversary cut in on the side detected")
            info = "CutInSideTracer"
            ego_fault = False
            i = [val for val in adv_side_cut_in if val][0]
            event_frame = all_frames_dict[i]
            event_frame["crashed"] = crashed
            self.all_stats["CutInSideTracer"].append(event_frame)

        elif any(ego_side_cut_in):
            ego_fault = True
            info = "EgoCutInSideTracer"
            print("Ego cut in on the side detected")
            i = [i for i, val in enumerate(ego_side_cut_in) if val][0]
            event_frame = all_frames_dict[i]
            event_frame["crashed"] = crashed
            self.all_stats["EgoCutInSideTracer"].append(event_frame)

        if any(ego_cut_out):
            i = [val for val in ego_cut_out if val][0]
            event_frame = all_frames_dict[i]
            event_frame["crashed"] = crashed
            self.all_stats["EgoCutOutTracer"].append(event_frame)

        print(f"Ego fault: {ego_fault}, Info: {info}")

        return ego_fault, info, metadata
