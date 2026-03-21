from __future__ import annotations

from .config import BaselineConfig, OutputConfig
from .train_baseline import train_multiseed_baselines


if __name__ == "__main__":
    baseline_cfg = BaselineConfig()
    output_cfg = OutputConfig()
    result = train_multiseed_baselines(baseline_cfg, output_cfg)
    print(result)
