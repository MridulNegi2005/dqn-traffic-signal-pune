"""Gymnasium environment: one Pune-style signalised crossing in SUMO."""
import os
import tempfile
import uuid
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

import config as C
from demand import arm_rates, load_counts, write_routes

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


class PuneCrossingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dates=None, episode_seconds=3600, random_start=True,
                 net_file=C.NET_FILE, scale=1.0, tripinfo=None, controller="agent",
                 seed=None, sublane=True, junction=C.TRAIN_JUNCTION, scale_range=None):
        super().__init__()
        self.dates = dates or C.TRAIN_DATES
        self.episode_seconds = episode_seconds
        self.random_start = random_start
        self.net_file = Path(net_file)
        self.scale = scale
        self.scale_range = scale_range
        self.junction = junction
        self.tripinfo = tripinfo
        self.controller = controller
        self.sublane = sublane
        self.counts = load_counts(junction)
        self.rates = {d: arm_rates(self.counts, d, junction) for d in self.dates}
        self.tmp = Path(tempfile.gettempdir()) / f"pune_rl_{uuid.uuid4().hex[:8]}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.sumo_seed = 0
        self._running = False

        self._read_phases()
        n_lanes = len(self.lanes)
        self.action_space = spaces.Discrete(len(self.greens))
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(2 * n_lanes + len(self.greens) + 2,), dtype=np.float32)

    def _read_phases(self):
        self._start_sumo(routes=None)
        tls = sumo_api.trafficlight.getIDList()[0]
        self.tls = tls
        logic = sumo_api.trafficlight.getAllProgramLogics(tls)[0]
        self.greens = [p.state for p in logic.phases
                       if "y" not in p.state and ("G" in p.state or "g" in p.state)]
        self.lanes = list(dict.fromkeys(sumo_api.trafficlight.getControlledLanes(tls)))
        self.lane_cap = {l: sumo_api.lane.getLength(l) / 7.5 for l in self.lanes}
        self._close()

    def _yellow(self, a, b):
        s, t = self.greens[a], self.greens[b]
        return "".join("y" if s[i] in "Gg" and t[i] not in "Gg" else s[i] for i in range(len(s)))

    def _start_sumo(self, routes):
        cmd = [sumo_binary(), "-n", str(self.net_file), "-a", str(C.VTYPE_FILE),
               "--step-length", "1", "--no-warnings", "--no-step-log",
               "--time-to-teleport", "300", "--waiting-time-memory", "10000",
               "--seed", str(self.sumo_seed if routes else 0)]
        if routes:
            cmd += ["-r", str(routes)]
        if self.sublane:
            cmd += ["--lateral-resolution", "0.8"]
        if self.tripinfo:
            cmd += ["--tripinfo-output", str(self.tripinfo),
                    "--tripinfo-output.write-unfinished",
                    "--device.emissions.probability", "1"]
        sumo_api.start(cmd)
        self._running = True

    def _close(self):
        if self._running:
            sumo_api.close()
            self._running = False

    # ---- episode -----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self._close()
        date = options.get("date") or self.dates[self.np_random.integers(len(self.dates))]
        day0 = pd.Timestamp(f"{date} {C.DAY_START}")
        if self.random_start:
            span = C.DAY_SECONDS - self.episode_seconds
            offset = int(self.np_random.integers(0, span // 300 + 1)) * 300
        else:
            offset = 0
        self.start = day0 + pd.Timedelta(seconds=offset)
        rates = self.rates.get(date)
        if rates is None:
            rates = arm_rates(self.counts, date, self.junction)
        routes = self.tmp / "routes.rou.xml"
        scale = self.scale
        if self.scale_range:
            scale = float(self.np_random.uniform(*self.scale_range))
        route_seed = options.get("route_seed")
        if route_seed is None:
            route_seed = int(self.np_random.integers(1e9))
        self.n_vehicles = write_routes(rates, self.start, self.episode_seconds, routes,
                                       seed=route_seed, scale=scale)
        self.sumo_seed = route_seed % 1_000_000
        self._start_sumo(routes)
        self.phase = 0
        self.elapsed = 0
        if self.controller == "agent":
            sumo_api.trafficlight.setRedYellowGreenState(self.tls, self.greens[0])
        self.last_wait = 0.0
        self.date = date
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if action == self.phase and self.elapsed >= C.MAX_GREEN:
            action = (self.phase + 1) % len(self.greens)
        if action != self.phase and self.elapsed >= C.MIN_GREEN:
            sumo_api.trafficlight.setRedYellowGreenState(self.tls, self._yellow(self.phase, action))
            self._sim(C.YELLOW)
            self.phase, self.elapsed = action, 0
        sumo_api.trafficlight.setRedYellowGreenState(self.tls, self.greens[self.phase])
        self._sim(C.DELTA_T)
        self.elapsed += C.DELTA_T

        wait = self._pcu_wait()
        reward = (self.last_wait - wait) / 100.0
        self.last_wait = wait
        done = sumo_api.simulation.getTime() >= self.episode_seconds
        return self._obs(), reward, done, False, {}

    def _sim(self, seconds):
        for _ in range(seconds):
            sumo_api.simulationStep()

    # ---- observation and reward -------------------------------------------
    def _pcu(self, veh):
        return C.PCU.get(sumo_api.vehicle.getTypeID(veh), 1.0)

    def _pcu_wait(self):
        total = 0.0
        for lane in self.lanes:
            for v in sumo_api.lane.getLastStepVehicleIDs(lane):
                total += self._pcu(v) * sumo_api.vehicle.getAccumulatedWaitingTime(v)
        return total

    def _obs(self):
        halt, occ = [], []
        for lane in self.lanes:
            pcu_all = pcu_halt = 0.0
            for v in sumo_api.lane.getLastStepVehicleIDs(lane):
                p = self._pcu(v)
                pcu_all += p
                if sumo_api.vehicle.getSpeed(v) < 0.1:
                    pcu_halt += p
            cap = self.lane_cap[lane]
            halt.append(min(pcu_halt / cap, 1.0))
            occ.append(min(pcu_all / cap, 1.0))
        phase = np.zeros(len(self.greens))
        phase[self.phase] = 1.0
        extra = [float(self.elapsed >= C.MIN_GREEN), min(self.elapsed / C.MAX_GREEN, 1.0)]
        return np.array(halt + occ + list(phase) + extra, dtype=np.float32)

    # ---- baselines ----------------------------------------------------------
    def run_uncontrolled(self, options=None):
        """Let SUMO's own program (fixed or actuated, from the net file) run the light."""
        self.controller = "sumo"
        self.reset(options=options)
        while sumo_api.simulation.getTime() < self.episode_seconds:
            sumo_api.simulationStep()
        self._close()

    def close(self):
        self._close()
