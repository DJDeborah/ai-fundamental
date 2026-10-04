"""Deterministic long-horizon arithmetic environment with an exact verifier."""
import random
import re


class ArithmeticChainEnv:
    def __init__(self, seed: int, horizon: int = 8):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        rng = random.Random(seed)
        self.start = rng.randrange(-5, 6)
        self.deltas = [rng.choice([*range(-9, 0), *range(1, 10)]) for _ in range(horizon)]
        self.targets = []
        value = self.start
        for delta in self.deltas:
            value += delta
            self.targets.append(value)
        self.horizon = horizon
        self.value = self.start
        self.index = 0
        self.history = []
        self.done = False
        self.success = False

    def observation(self) -> str:
        if self.done:
            raise RuntimeError("episode ended")
        recent = ", ".join(self.history[-2:]) or "none"
        return (f"Step {self.index+1}/{self.horizon}. Current value: {self.value}. "
                f"Next checkpoint: {self.targets[self.index]}. Recent actions: {recent}. "
                "Reply with exactly ADD <integer>.")

    def oracle_action(self) -> str:
        return f"ADD {self.deltas[self.index]}"

    def step(self, action: str):
        if self.done:
            raise RuntimeError("episode ended")
        match = re.fullmatch(r"ADD ([+-]?(?:0|[1-9][0-9]*))", action.strip())
        delta = int(match.group(1)) if match else None
        valid = delta is not None and -9 <= delta <= 9 and self.value + delta == self.targets[self.index]
        self.history.append(action.strip())
        if valid:
            self.value += delta
            self.index += 1
            self.done = self.index == self.horizon
            self.success = self.done
        else:
            self.done = True
            self.success = False
        return {"valid": valid, "done": self.done, "success": self.success,
                "reward": (1.0 / self.horizon + (1.0 if self.success else 0.0)) if valid else 0.0}
