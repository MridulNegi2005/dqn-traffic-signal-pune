"""Gymnasium environment: one Pune-style signalised crossing in SUMO.

The signal serves one road at a time (split phasing), which is common at Pune
junctions. Each road has a left-turn lane, a straight lane and a straight/right lane.
With free_left=True the left-turn lanes act as slip lanes and may go on red.
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


class PuneCrossingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dates=None, episode_seconds=3600, random_start=True, scale=1.0,
                 tripinfo=None, controller="agent", free_left=False, sublane=True,
                 junction=C.TRAIN_JUNCTION, scale_range=None):
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
        self.counts = load_counts(junction)
        self.rates = {d: arm_rates(self.counts, d, junction) for d in self.dates}
        self.tmp = Path(tempfile.gettempdir()) / f"pune_rl_{uuid.uuid4().hex[:8]}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.sumo_seed = 0
        self._running = False

        self._build_phases()
        self.action_space = spaces.Discrete(len(C.ARMS))
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(2 * len(self.lanes) + len(C.ARMS) + 2,), dtype=np.float32)

    # ---- signal plan ----------------------------------------------------------
    def _build_phases(self):
        self._start_sumo(routes=None)
        self.tls = sumo_api.trafficlight.getIDList()[0]
        links = sumo_api.trafficlight.getControlledLinks(self.tls)
        self.lanes = list(dict.fromkeys(sumo_api.trafficlight.getControlledLanes(self.tls)))
        self.lane_cap = {l: sumo_api.lane.getLength(l) / 7.5 for l in self.lanes}
        self._close()

        arm_of, is_left = [], []
        for link in links:
            src, dst = link[0][0], link[0][1]
            a, b = src.split("_")[0], dst.split("_")[0]
            arm_of.append(a)
            is_left.append(b == LEFT_OF[a])

        def state(active=None, colour="G"):
            s = []
            for a, left in zip(arm_of, is_left):
                if a == active:
                    s.append(colour)
                elif left and self.free_left:
                    s.append("g")
                else:
                    s.append("r")
            return "".join(s)

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
        self.last_wait = 0.0
        if self.controller == "agent":
            sumo_api.trafficlight.setRedYellowGreenState(self.tls, self.greens[0])
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if action == self.phase and self.elapsed >= C.MAX_GREEN:
            action = (self.phase + 1) % len(C.ARMS)
        if action != self.phase and self.elapsed >= C.MIN_GREEN:
            sumo_api.trafficlight.setRedYellowGreenState(self.tls, self.yellows[self.phase])
            self._sim(C.YELLOW)
            sumo_api.trafficlight.setRedYellowGreenState(self.tls, self.all_red)
            self._sim(C.ALL_RED)
            self.phase, self.elapsed = action, 0
        sumo_api.trafficlight.setRedYellowGreenState(self.tls, self.greens[self.phase])
        self._sim(C.DELTA_T)
        self.elapsed += C.DELTA_T

        wait = self._pcu_wait()
        reward = (self.last_wait - wait) / 100.0
        self.last_wait = wait
        done = sumo_api.simulation.getTime() >= self.episode_seconds
        return self._obs(), reward, done, False, {}

    def run_baseline(self, options=None):
        """Run the whole episode under the fixed or actuated program."""
        self.reset(options=options)
        while sumo_api.simulation.getTime() < self.episode_seconds:
            sumo_api.simulationStep()
        self._close()

    def _sim(self, seconds):
        for _ in range(seconds):
            sumo_api.simulationStep()

    # ---- observation and reward -----------------------------------------------
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
        phase = np.zeros(len(C.ARMS))
        phase[self.phase] = 1.0
        extra = [float(self.elapsed >= C.MIN_GREEN), min(self.elapsed / C.MAX_GREEN, 1.0)]
        return np.array(halt + occ + list(phase) + extra, dtype=np.float32)

    def close(self):
        self._close()
