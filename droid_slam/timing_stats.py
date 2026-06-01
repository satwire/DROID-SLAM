import json
import time
from collections import defaultdict


class TimingStats:
    def __init__(self):
        self._times: dict[str, list[float]] = defaultdict(list)
        self._pending: dict[str, float] = {}

    def start(self, section: str):
        self._pending[section] = time.perf_counter()

    def stop(self, section: str):
        if section in self._pending:
            self._times[section].append(
                time.perf_counter() - self._pending.pop(section)
            )

    def record(self, section: str, elapsed: float):
        self._times[section].append(elapsed)

    def summary(self) -> dict:
        out = {}
        for section in sorted(self._times):
            times = self._times[section]
            n = len(times)
            total = sum(times)
            out[section] = {
                "count": n,
                "total_s": round(total, 4),
                "mean_ms": round(total / n * 1000, 3) if n else 0,
                "min_ms": round(min(times) * 1000, 3) if times else 0,
                "max_ms": round(max(times) * 1000, 3) if times else 0,
            }
        return out

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        print(f"Timing stats saved to {path}.")
