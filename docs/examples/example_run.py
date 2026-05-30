"""
Minimal example script showing how to import and create the Husky environment.
This is a lightweight, non-executing example — adapt to your project layout.
"""

try:
    # adjust import path if running from docs/examples directory
    import sys
    from pathlib import Path
    import importlib.util

    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    husky_path = project_root / "husky_env.py"
    if husky_path.exists():
        spec = importlib.util.spec_from_file_location("husky_env", str(husky_path))
        husky = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(husky)
        HuskyEnv = getattr(husky, "HuskyEnv", None)
    else:
        HuskyEnv = None
except Exception:
    HuskyEnv = None


def main():
    if HuskyEnv is None:
        print("Could not import HuskyEnv automatically. Run this from the project root or adjust PYTHONPATH.")
        return

    env = HuskyEnv()
    obs = env.reset()
    print('Environment reset. Observation shape:', getattr(obs, 'shape', type(obs)))


if __name__ == '__main__':
    main()
