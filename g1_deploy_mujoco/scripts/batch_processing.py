#!/usr/bin/env python3
# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import glob
import shutil
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from copy import deepcopy


# ---------- Helper Functions ----------
def _policy_actions(policy, obs: torch.Tensor):
    """
    Unified way to extract actions from a policy:
    1) Prefer policy.act(obs, deterministic=True)
    2) Otherwise try policy.forward(obs)
    3) Handle dict or tuple outputs by extracting the action tensor
    """
    if hasattr(policy, "act"):
        out = policy.act(obs, deterministic=True)
    elif hasattr(policy, "forward"):
        try:
            out = policy.forward(obs)
        except TypeError:
            out = policy.forward()
    else:
        raise RuntimeError("Policy has neither .act(...) nor .forward(...).")

    if isinstance(out, dict):
        if "actions" in out:
            out = out["actions"]
        else:
            for v in out.values():
                if torch.is_tensor(v):
                    out = v
                    break
    elif isinstance(out, tuple):
        for v in out:
            if torch.is_tensor(v):
                out = v
                break

    if not torch.is_tensor(out):
        raise RuntimeError(f"Policy output is not a Tensor: type={type(out)}")
    return out


def _collect_checkpoints(inputs):
    """Collect checkpoints from files / directories / wildcard patterns."""
    results = []
    patterns = ("*.pt", "*.pth")
    for item in inputs:
        p = Path(item).expanduser()
        if any(ch in item for ch in "*?[]"):  # glob pattern
            results.extend([Path(x) for x in glob.glob(item)])
        elif p.is_dir():
            for pat in patterns:
                results.extend(p.rglob(pat))
        elif p.is_file():
            results.append(p)
        else:
            print(f"[WARN] Path does not exist or does not match: {item}")
    return sorted({x.resolve() for x in results if x.is_file()})


def _apply_normalizer(normalizer, x):
    """Apply observation normalizer if available."""
    if normalizer is None:
        return x
    if hasattr(normalizer, "normalize"):
        return normalizer.normalize(x)
    if callable(normalizer):
        return normalizer(x)
    return x


def load_ckpt_smart(path: str | Path, device: str | torch.device):
    """
    Smart loader for either:
      - state_dict-like checkpoints (training dict or pure state_dict)
      - TorchScript archives (ScriptModule)
    Returns:
      {"kind":"state_dict", "state_dict":..., "extra":...}
      or
      {"kind":"script", "module":ScriptModule}
    """
    p = str(path)

    # 1) Try state_dict with weights_only=True (PyTorch 2.6 default)
    try:
        obj = torch.load(p, map_location=device, weights_only=True)
        if isinstance(obj, dict):
            if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
                return {"kind": "state_dict", "state_dict": obj["model_state_dict"], "extra": obj}
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return {"kind": "state_dict", "state_dict": obj["state_dict"], "extra": obj}
            # Some checkpoints are pure state_dict
            if all(isinstance(k, str) for k in obj.keys()):
                return {"kind": "state_dict", "state_dict": obj, "extra": obj}
    except RuntimeError as e:
        # TorchScript archive + weights_only=True will land here
        if "TorchScript archives" not in str(e):
            raise

    # 2) Try as TorchScript
    try:
        script_mod = torch.jit.load(p, map_location=device)
        return {"kind": "script", "module": script_mod}
    except Exception:
        pass

    # 3) Fallback state_dict with weights_only=False (only for trusted sources)
    obj = torch.load(p, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return {"kind": "state_dict", "state_dict": obj["model_state_dict"], "extra": obj}
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return {"kind": "state_dict", "state_dict": obj["state_dict"], "extra": obj}
        if all(isinstance(k, str) for k in obj.keys()):
            return {"kind": "state_dict", "state_dict": obj, "extra": obj}

    raise ValueError(f"Unrecognized checkpoint format: {path}")


# ---------- Export Implementation ----------
def export_policy_as_jit(policy, normalizer, path, filename="policy.pt", obs_dim: int = None, device="cpu"):
    assert obs_dim is not None, "export_policy_as_jit requires obs_dim"

    class ExportPolicy(torch.nn.Module):
        def __init__(self, policy, normalizer):
            super().__init__()
            self.policy = policy
            self.normalizer = normalizer

        def forward(self, observations):
            observations = _apply_normalizer(self.normalizer, observations)
            actions = _policy_actions(self.policy, observations)
            return actions

    export_policy = ExportPolicy(policy, normalizer).eval()
    example_obs = torch.randn(1, obs_dim, dtype=torch.float32, device=device)
    os.makedirs(path, exist_ok=True)
    with torch.no_grad():
        traced = torch.jit.trace(export_policy, example_obs)
    traced.save(os.path.join(path, filename))


def export_policy_as_onnx(policy, normalizer, path, filename="policy.onnx",
                          input_names=("observations",), output_names=("actions",),
                          obs_dim: int = None, device="cpu"):
    assert obs_dim is not None, "export_policy_as_onnx requires obs_dim"

    class ExportPolicy(torch.nn.Module):
        def __init__(self, policy, normalizer):
            super().__init__()
            self.policy = policy
            self.normalizer = normalizer

        def forward(self, observations):
            observations = _apply_normalizer(self.normalizer, observations)
            actions = _policy_actions(self.policy, observations)
            return actions

    export_policy = ExportPolicy(policy, normalizer).eval()
    example_obs = torch.randn(1, obs_dim, dtype=torch.float32, device=device)
    os.makedirs(path, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            export_policy,
            example_obs,
            os.path.join(path, filename),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes={input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}},
            opset_version=12,
        )


# ---------- Minimal Dummy Env ----------
class DummyEnv:
    """Minimal Dummy Environment: only provides interfaces for OnPolicyRunner init."""
    def __init__(self, obs_dim, action_dim, device):
        self.obs_dim = int(obs_dim)
        self.privileged_obs_dim = 15
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.num_envs = 1

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.privileged_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.privileged_obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        class _U: pass
        self.unwrapped = _U()
        self.unwrapped.device = self.device

        self.num_obs = self.obs_dim
        self.num_privileged_obs = self.privileged_obs_dim
        self.num_actions = self.action_dim

    def get_observations(self):
        obs = torch.zeros(self.num_envs, self.obs_dim, device=self.device, dtype=torch.float32)
        privileged_obs = torch.zeros(self.num_envs, self.privileged_obs_dim, device=self.device, dtype=torch.float32)
        extras = {
            "observations": {},                  # no critic-specific obs
            "privileged_observations": privileged_obs,
            "rew_terms": {},                     # reward terms placeholder
        }
        return obs, extras

    def reset(self): raise RuntimeError("DummyEnv.reset() should not be called")
    def step(self, a): raise RuntimeError("DummyEnv.step() should not be called")


# ---------- CLI ----------
def build_argparser():
    p = argparse.ArgumentParser("Batch export RSL-RL checkpoints to JIT/ONNX (no IsaacSim needed)")

    p.add_argument("--input_path", nargs="*", default=["example.pt"],
                   help="Checkpoint file(s) / directory / wildcard, multiple allowed")
    p.add_argument("--output_path", type=str, default=".",
                   help="Optional: unified export root. Defaults to ckpt folder/exported_<stem>")

    p.add_argument("--obs_dim", type=int, default=480, help="Observation dimension (e.g. G1: 480)")
    p.add_argument("--action_dim", type=int, default=29, help="Action dimension (e.g. G1: 29)")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device to run on, e.g. cpu / cuda:0 / cuda:1")
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    ckpts = _collect_checkpoints(args.input_path)
    if not ckpts:
        print("[ERROR] No .pt/.pth checkpoints found")
        return
    print(f"[INFO] Found {len(ckpts)} checkpoints; device={args.device}")

    # Minimal runner config (avoid IsaacLab deps)
    agent_cfg = {
        "device": args.device,
        "clip_actions": True,
        "num_steps_per_env": 16,
        "max_iterations": 1,
        "save_interval": 1000000,
        "experiment_name": "export_only",
        "runner_log_interval": 1000000,
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }

    env = DummyEnv(args.obs_dim, args.action_dim, device=args.device)

    for ckpt in ckpts:
        print(f"\n[INFO] Loading: {ckpt}")
        cfg = deepcopy(agent_cfg)
        runner = OnPolicyRunner(env, cfg, log_dir=None, device=args.device)

        info = load_ckpt_smart(ckpt, args.device)

        ckpt_name = ckpt.stem
        if ckpt_name.startswith("model_"):
            model_number = ckpt_name.replace("model_", "")
            jit_filename = f"policy_{model_number}.pt"
            onnx_filename = f"policy_{model_number}.onnx"
        else:
            jit_filename = "policy.pt"
            onnx_filename = "policy.onnx"

        # 这里直接用固定的 exported 文件夹
        export_dir = Path(args.output_path).expanduser().resolve() / "exported"
        export_dir.mkdir(parents=True, exist_ok=True)


        if info["kind"] == "state_dict":
            # Load weights into fresh policy
            model_state_dict = info["state_dict"]
            current_state_dict = runner.alg.policy.state_dict()

            filtered_state_dict = {
                k: v for k, v in model_state_dict.items()
                if k in current_state_dict and getattr(v, "shape", None) == getattr(current_state_dict[k], "shape", None)
            }
            missing = [k for k in current_state_dict.keys() if k not in filtered_state_dict]
            if missing:
                print(f"[WARN] {len(missing)} keys missing or shape-mismatch; loading filtered subset.")

            runner.alg.policy.load_state_dict(filtered_state_dict, strict=False)

            # Extract policy module (compatibility across versions)
            try:
                policy_nn = runner.alg.policy
            except AttributeError:
                policy_nn = runner.alg.actor_critic

            normalizer = getattr(runner, "obs_normalizer", None) or getattr(runner.alg, "obs_normalizer", None)

            # Export JIT
            try:
                export_policy_as_jit(policy_nn.actor, normalizer, path=str(export_dir),
                                     filename=jit_filename, obs_dim=args.obs_dim, device=args.device)
                print(f"[OK] JIT:   {export_dir/jit_filename}")
            except Exception as e:
                print(f"[FAIL] JIT: {e}")

            # Export ONNX
            try:
                export_policy_as_onnx(policy_nn.actor, normalizer, path=str(export_dir),
                                      filename=onnx_filename, obs_dim=args.obs_dim, device=args.device)
                print(f"[OK] ONNX:  {export_dir/onnx_filename}")
            except Exception as e:
                print(f"[FAIL] ONNX: {e}")

        elif info["kind"] == "script":
            # Already TorchScript module
            script_mod = info["module"].eval()
            # Save JIT as-is
            try:
                target = export_dir / jit_filename
                # Prefer re-save to ensure device-neutral
                script_mod.save(str(target))
                print(f"[OK] JIT (from TorchScript): {target}")
            except Exception as e:
                # Fallback: copy original file
                print(f"[WARN] Re-saving ScriptModule failed ({e}); copying original file.")
                shutil.copy2(str(ckpt), str(export_dir / jit_filename))
                print(f"[OK] JIT (copied): {export_dir / jit_filename}")

            # ONNX: usually not supported from ScriptModule
            print("[INFO] Skipping ONNX export for TorchScript archive. "
                  "Use a training state_dict checkpoint to export ONNX.")

    print("\n[DONE] All checkpoints processed.")


if __name__ == "__main__":
    main()
