"""Gymnasium environment: one Pune-style signalised crossing in SUMO.

The signal serves one road at a time (split phasing), which is common at Pune
junctions. Each road has a left-turn lane, a straight lane and a straight/right lane.
With free_left=True the left-turn lanes act as slip lanes and may go on red.

Options
  obs:          "camera" sees only the last CAMERA_RANGE metres of each lane; "full" sees the lane.
  reward:       "combined" (queue, spill-back, throughput, switch cost, longest wait) or
                "wait" (drop in PCU-weighted accumulated waiting time).
  action_mode:  "choose" picks the road to serve; "keepswitch" keeps the green or switches to
                the road the camera rule ranks highest.
"""
import os
import tempfile
import uuid
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

import config as C
from demand import LEFT_OF, arm_rates, load_counts, write_routes

if os.environ.get("USE_TRACI") == "1":
    import traci as sumo_api
else:
    try:
        import libsumo as sumo_api
    except ImportError:
        import traci as sumo_api


def sumo_binary() -> str:
    try:
        import sumo
        return str(Path(sumo.SUMO_HOME) / "bin" / "sumo")
    except ImportError:
        return "sumo"


def pcu_of(veh_id: str) -> float:
    return C.PCU.get(veh_id.split(".")[0], 1.0)


class PuneCrossingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dates=None, episode_seconds=3600, random_start=True, scale=1.0,
                 tripinfo=None, controller="agent", free_left=False, sublane=True,
                 junction=C.TRAIN_JUNCTION, scale_range=None, delta_t=C.DELTA_T,
                 obs="camera", reward="combined", action_mode="choose"):
        super().__init__()
        assert controller in ("agent", "fixed", "actuated")
        self.dates = dates or C.TRAIN_DATES
        self.episode_seconds = episode_seconds
        self.random_start = random_start
        self.scale = scale
        self.scale_range = scale_range
        self.junction = junction
        self.tripinfo = tripinfo
        self.controller = controller
        self.free_left = free_left
        self.sublane = sublane
        self.delta_t = int(delta_t)
        self.obs_mode = obs
        self.reward_mode = reward
        self.action_mode = action_mode
        self.counts = load_counts(junction)
        self.rates = {d: arm_rates(self.counts, d, junction) for d in self.dates}
        self.tmp = Path(tempfile.gettempdir()) / f"pune_rl_{uuid.uuid4().hex[:8]}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.sumo_seed = 0
        self._running = False

        self._build_phases()
        n_act = len(C.ARMS) if action_mode == "choose" else 2
        self.action_space = spaces.Discrete(n_act)
        n_obs = 2 * len(self.lanes) + 2 * len(C.ARMS) + 2
        self.observation_space = spaces.Box(0.0, 1.0, shape=(n_obs,), dtype=np.float32)

    # ---- signal plan ----------------------------------------------------------
    def _build_phases(self):
        self._start_sumo(routes=None)
        self.tls = sumo_api.trafficlight.getIDList()[0]
        links = sumo_api.trafficlight.getControlledLinks(self.tls)
        self.lanes = list(dict.fromkeys(sumo_api.trafficlight.getControlledLanes(self.tls)))
        self.lane_len = {l: sumo_api.lane.getLength(l) for l in self.lanes}
        self._close()
        self.lane_arm = {l: l.split("_")[0] for l in self.lanes}
        view = {l: min(self.lane_len[l], C.CAMERA_RANGE) if self.obs_mode == "camera"
                else self.lane_len[l] for l in self.lanes}
        self.view_from = {l: self.lane_len[l] - view[l] for l in self.lanes}
        self.view_cap = {l: view[l] / 7.5 for l in self.lanes}

        arm_of, is_left = [], []
        for link in links:
            a, b = link[0][0].split("_")[0], link[0][1].split("_")[0]
            arm_of.append(a)
            is_left.append(b == LEFT_OF[a])

        def state(active=None, colour="G"):
            out = []
            for a, left in zip(arm_of, is_left):
                if a == active:
                    out.append(colour)
                elif left and self.free_left:
                    out.append("g")
                else:
                    out.append("r")
            return "".join(out)

        self.greens = [state(a, "G") for a in C.ARMS]
        self.yellows = [state(a, "y") for a in C.ARMS]
        self.all_red = state(None)
        if self.controller != "agent":
            self.tll_file = self._write_program()

    def _write_program(self):
        actuated = self.controller == "actuated"
        kind = "actuated" if actuated else "static"
        lines = ["<additional>", f'  <tlLogic id="{self.tls}" type="{kind}" programID="pune" offset="0">']
        for g, y in zip(self.greens, self.yellows):
            if actuated:
                lines.append(f'    <phase duration="{C.FIXED_GREEN}" minDur="{C.MIN_GREEN}" '
                             f'maxDur="{C.MAX_GREEN}" state="{g}"/>')
            else:
                lines.append(f'    <phase duration="{C.FIXED_GREEN}" state="{g}"/>')
            lines.append(f'    <phase duration="{C.YELLOW}" state="{y}"/>')
            lines.append(f'    <phase duration="{C.ALL_RED}" state="{self.all_red}"/>')
        lines += ["  </tlLogic>", "</additional>"]
        path = self.tmp / f"{kind}.tll.xml"
        path.write_text("\n".join(lines))
        return path

    # ---- SUMO process ---------------------------------------------------------
    def _start_sumo(self, routes):
        adds = [str(C.VTYPE_FILE)]
        if routes and self.controller != "agent":
            adds.append(str(self.tll_file))
        cmd = [sumo_binary(), "-n", str(C.NET_FILE), "-a", ",".join(adds),
               "--step-length", "1", "--no-warnings", "--no-step-log",
               "--time-to-teleport", "300", "--waiting-time-memory", "10000",
               "--seed", str(self.sumo_seed if routes else 0)]
        if routes:
            cmd += ["-r", str(routes)]
        if self.sublane:
            cmd += ["--lateral-resolution", "0.8"]
        if self.tripinfo and routes:
            cmd += ["--tripinfo-output", str(self.tripinfo),
                    "--tripinfo-output.write-unfinished",
                    "--device.emissions.probability", "1"]
        sumo_api.start(cmd)
        self._running = True

    def _close(self):
        if self._running:
            sumo_api.close()
            self._running = False

    # ---- episode --------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self._close()
        date = options.get("date") or self.dates[self.np_random.integers(len(self.dates))]
        day0 = pd.Timestamp(f"{date} {C.DAY_START}")
        offset = 0
        if self.random_start:
            span = C.DAY_SECONDS - self.episode_seconds
            offset = int(self.np_random.integers(0, span // 300 + 1)) * 300
        self.start = day0 + pd.Timedelta(seconds=offset)
        rates = self.rates.get(date)
        if rates is None:
            rates = arm_rates(self.counts, date, self.junction)
        scale = self.scale
        if self.scale_range:
            scale = float(self.np_random.uniform(*self.scale_range))
        route_seed = options.get("route_seed")
        if route_seed is None:
            route_seed = int(self.np_random.integers(1e9))
        routes = self.tmp / "routes.rou.xml"
        self.n_vehicles = write_routes(rates, self.start, self.episode_seconds, routes,
                                       seed=route_seed, scale=scale)
        self.sumo_seed = route_seed % 1_000_000
        self._start_sumo(routes)
        self.phase, self.elapsed = 0, 0
        self.green_log = []
        self.on_lanes = set()
        self.last_wait = 0.0
        if self.controller == "agent":
            sumo_api.trafficlight.setRedYellowGreenState(self.tls, self.greens[0])
        self._scan()
        return self._obs(), {}

    def step(self, action):
        target = self._target(int(action))
        switched = False
        if target == self.phase and self.elapsed >= C.MAX_GREEN:
            target = self._best_other()
        if target != self.phase and self.elapsed >= C.MIN_GREEN:
            self.green_log.append((C.ARMS[self.phase], self.elapsed))
            sumo_api.trafficlight.setRedYellowGreenState(self.tls, self.yellows[self.phase])
            cost = self._sim(C.YELLOW)
            sumo_api.trafficlight.setRedYellowGreenState(self.tls, self.all_red)
            cost += self._sim(C.ALL_RED)
            self.phase, self.elapsed, switched = target, 0, True
        else:
            cost = 0.0
        sumo_api.trafficlight.setRedYellowGreenState(self.tls, self.greens[self.phase])
        cost += self._sim(self.delta_t)
        self.elapsed += self.delta_t

        if self.reward_mode == "combined":
            reward = -cost + C.W_THROUGHPUT * self.crossed_pcu - C.W_SWITCH * switched
        else:
            wait = self._pcu_wait()
            reward = (self.last_wait - wait) / 100.0
            self.last_wait = wait
        done = sumo_api.simulation.getTime() >= self.episode_seconds
        if done:
            self.green_log.append((C.ARMS[self.phase], self.elapsed))
        return self._obs(), float(reward), done, False, {}

    def _target(self, action):
        if self.action_mode == "choose":
            return action
        return self.phase if action == 0 else self._best_other()

    # ---- simulation and per-second measurement --------------------------------
    def _sim(self, seconds):
        """Advance the simulation; return the integrated per-second cost of the combined reward."""
        cost = 0.0
        for _ in range(seconds):
            sumo_api.simulationStep()
            self._scan()
            cost += (C.W_QUEUE * self.queue_pcu + C.W_BACKLOG * self.backlog_pcu
                     + C.W_MAXWAIT * self.max_wait)
        return cost

    def _scan(self):
        """Read every incoming lane once per simulated second."""
        on = set()
        self.queue_pcu = 0.0
        self.max_wait = 0.0
        self.lane_view = {}
        arm_wait = {a: 0.0 for a in C.ARMS}
        for lane in self.lanes:
            halt_v = all_v = 0.0
            for v in sumo_api.lane.getLastStepVehicleIDs(lane):
                on.add(v)
                p = pcu_of(v)
                stopped = sumo_api.vehicle.getSpeed(v) < 0.1
                w = sumo_api.vehicle.getAccumulatedWaitingTime(v)
                if stopped:
                    self.queue_pcu += p
                self.max_wait = max(self.max_wait, w)
                if sumo_api.vehicle.getLanePosition(v) >= self.view_from[lane]:
                    all_v += p
                    if stopped:
                        halt_v += p
                    arm = self.lane_arm[lane]
                    arm_wait[arm] = max(arm_wait[arm], w)
            self.lane_view[lane] = (halt_v, all_v)
        self.arm_wait = arm_wait
        self.crossed_pcu = (getattr(self, "crossed_pcu", 0.0)
                            + sum(pcu_of(v) for v in self.on_lanes - on))
        self.on_lanes = on
        self.backlog_pcu = sum(pcu_of(v) for v in sumo_api.simulation.getPendingVehicles())

    def _pcu_wait(self):
        total = 0.0
        for lane in self.lanes:
            for v in sumo_api.lane.getLastStepVehicleIDs(lane):
                total += pcu_of(v) * sumo_api.vehicle.getAccumulatedWaitingTime(v)
        return total

    def _obs(self):
        self.crossed_pcu = 0.0
        halt = [min(self.lane_view[l][0] / self.view_cap[l], 1.0) for l in self.lanes]
        occ = [min(self.lane_view[l][1] / self.view_cap[l], 1.0) for l in self.lanes]
        wait = [min(self.arm_wait[a] / C.MAX_WAIT_NORM, 1.0) for a in C.ARMS]
        phase = [1.0 if i == self.phase else 0.0 for i in range(len(C.ARMS))]
        extra = [float(self.elapsed >= C.MIN_GREEN), min(self.elapsed / C.MAX_GREEN, 1.0)]
        return np.array(halt + occ + wait + phase + extra, dtype=np.float32)

    # ---- camera rule (baseline and teacher) -----------------------------------
    def _arm_score(self, arm):
        q = sum(self.lane_view[l][0] for l in self.lanes if self.lane_arm[l] == arm)
        return q + 0.05 * self.arm_wait[arm]

    def _best_other(self):
        others = [i for i in range(len(C.ARMS)) if i != self.phase]
        return max(others, key=lambda i: self._arm_score(C.ARMS[i]))

    def _green_busy(self):
        """Vehicles still moving or queued in the last 30 m of the green road."""
        arm = C.ARMS[self.phase]
        for lane in self.lanes:
            if self.lane_arm[lane] != arm:
                continue
            for v in sumo_api.lane.getLastStepVehicleIDs(lane):
                if sumo_api.vehicle.getLanePosition(v) >= self.lane_len[lane] - 30.0:
                    return True
        return False

    def teacher_action(self):
        """Camera rule: hold the green while vehicles reach the stop line, then serve the
        road with the largest PCU queue and wait. Returned in this env's action format."""
        keep = self.elapsed < C.MIN_GREEN or (self._green_busy() and self.elapsed < C.MAX_GREEN)
        if not keep:
            best = self._best_other()
            if self._arm_score(C.ARMS[best]) <= 0 and self.elapsed < C.MAX_GREEN:
                keep = True
        if self.action_mode == "keepswitch":
            return 0 if keep else 1
        return self.phase if keep else self._best_other()

    def run_baseline(self, options=None):
        """Run the whole episode under the fixed or actuated program."""
        self.reset(options=options)
        while sumo_api.simulation.getTime() < self.episode_seconds:
            sumo_api.simulationStep()
        self._close()

    def close(self):
        self._close()
