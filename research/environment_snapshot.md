# WP0 environment snapshot

Captured: 2026-07-27 13:57:27 -03:00

## Host and runtime

| Item | Observed value |
|---|---|
| Kernel and platform | Linux WSL2, kernel `6.6.114.1-microsoft-standard-WSL2`, `x86_64` |
| CPU | 13th Gen Intel(R) Core(TM) i9-13900H |
| Logical CPUs | 20 |
| Physical cores | 10 |
| Threads per core | 2 |
| RAM | 15 GiB total |
| RAM available at capture | 8.4 GiB |
| Swap | 4.0 GiB total, 0 B used |
| Python | 3.12.3 |
| R | 4.3.3, x86_64-pc-linux-gnu |
| Shell | zsh |

## Long-running or competing work observed

The process snapshot showed an existing simulation command with PID `499326`,
`python3 -u experiments/run_simulations.py --n_reps 500 --sample_siz...`, and eight visible loky worker processes with PIDs `499387` through `499394`. These workers were each reported at approximately 108 percent CPU in the snapshot. Several editor and agent processes were also active.

## Execution consequence

Do not launch the WP9 simulation sweep from WP0. Before any later experiment, repeat the CPU and memory snapshot. The default local cap is eight workers only when the machine is otherwise idle; competing work or low available RAM requires a lower cap. Keep expected memory below 12 GiB and avoid relying on swap.

## Reproducibility note

This snapshot records the runtime versions available at WP0. A later clean-run snapshot must also record package versions and the project lockfile before final experiments.

