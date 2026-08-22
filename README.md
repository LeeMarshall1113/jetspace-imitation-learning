# jetspace-imitation-learning

Imitation learning + reinforcement learning on a **frozen JEPA latent world model**.

Teleoperated demonstrations train an action-conditioned predictor on top of a
frozen V-JEPA 2 encoder; a policy is then behavior-cloned from the demos and
improved by RL *inside* that latent world model. The goal is a single policy that
transfers to held-out variations of a task without task-specific retraining.

> **Status:** M0, environment setup. Nothing is trained yet.
> Read [`REQUIREMENTS.md`](REQUIREMENTS.md) for scope, success gates and timeline.

## Quickstart

```bash
docker compose -f docker/compose.yaml --profile wsl2 run --rm dev-wsl
python scripts/check_env.py     # always run this first
```

Profiles: `wsl2` (Windows + Radeon), `linux` (native ROCm), `cpu`.
Full instructions and a failure-mode table: [`docs/setup.md`](docs/setup.md).

## Three things to know before reading the code

1. **The JEPA is frozen and pretrained.** We do not train a world model from
   scratch — that is a >1M video-hour, cluster-scale job. V-JEPA 2 is pretrained
   on internet video and V-JEPA 2-AC already learned action-conditioned prediction
   from <62 hours of robot teleop. We train the action head and the policy. This
   is what makes the project fit on one 16 GB consumer GPU.

2. **MuJoCo is the default simulator, not Isaac Sim.** Isaac Sim requires an
   NVIDIA RTX GPU (min RTX 4080) and cannot run on the Radeon dev box at all.
   `src/jetspace/envs/base.py` is the seam that keeps an Isaac backend possible on
   `feat/isaac-backend` without forking the repo.

3. **The world model reads RGB, not depth.** Depth is optional metadata only. See
   [`docs/architecture.md#sensing`](docs/architecture.md).

## Layout

```
docker/          ROCm image + compose profiles (wsl2 / linux / cpu)
scripts/         check_env.py and entry points
src/jetspace/
  envs/          RobotEnv abstraction + MuJoCo backend
  data/          teleop capture, dataset, latent caching
  models/        frozen encoder, action-conditioned head
  policies/      BC, latent-imagination RL
configs/         omegaconf configs
docs/            architecture, setup, references
```

## Branches

- `main` — environment setup; must always build and pass `check_env.py`.
- `dev` — integration branch; feature work merges here first.
- `feat/*` — feature branches (e.g. `feat/isaac-backend`).

## Documentation

| Doc | What it covers |
|-----|----------------|
| [`REQUIREMENTS.md`](REQUIREMENTS.md) | Success gates, compute budget, milestones, non-goals |
| [`docs/architecture.md`](docs/architecture.md) | Why frozen-encoder + latent RL, backend seam, sensing |
| [`docs/setup.md`](docs/setup.md) | WSL2/ROCm/Docker setup and failure modes |
| [`docs/references.md`](docs/references.md) | Source audit — **includes corrections to the original brief** |

## License

Apache-2.0. See [`LICENSE`](LICENSE).
