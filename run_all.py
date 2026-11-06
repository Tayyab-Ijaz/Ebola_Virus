#!/usr/bin/env python3
"""Single entry point for the CSBJ-D-26-00901 re-analysis.

  python run_all.py --stages all                  # every implemented stage, one run
  python run_all.py --stages item6,item7,item10a  # a subset (still one run directory, one log)
  python run_all.py --verify results_v2/<run_id>  # re-hash a finished run, detect splicing
  python run_all.py --list

One run = one directory results_v2/<run_id>/ containing run.log, config.used.json, manifest.json
(package + R versions, git commit/dirty flag, input and plan hashes) and one subfolder per stage.
A run directory is never reused and no output file is ever overwritten.
"""
import argparse
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evd.config import load_config  # noqa: E402
from evd.provenance import RunContext, r_versions, verify_run  # noqa: E402
import stages  # noqa: E402


def _power_state():
    """(on_ac, battery_percent) on Windows, (None, None) elsewhere. A run killed by a flat battery (4:36 am, 24 Oct 2026)
    is what aborted final_rerun_20261024, so the run refuses to start on battery power."""
    if sys.platform != "win32":
        return None, None

    class SPS(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte), ("res", ctypes.c_ubyte),
                    ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    s = SPS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return None, None
    return s.ac == 1, (None if s.pct == 255 else int(s.pct))


def _keep_awake():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--verify", default=None, metavar="RUN_DIR")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run; allowed only with the same --run-id, commit and config, clean tree")
    ap.add_argument("--allow-battery", action="store_true", help="do not refuse to start on battery power")
    a = ap.parse_args()

    if a.list:
        print("implemented:", ", ".join(stages.IMPLEMENTED))
        print("planned, not implemented yet:", ", ".join(stages.PENDING))
        return 0
    if a.verify:
        problems = verify_run(a.verify)
        print("OK: run is internally consistent" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        return 1 if problems else 0

    todo = stages.IMPLEMENTED if a.stages == "all" else [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in todo if s not in stages.IMPLEMENTED]
    if bad:
        print(f"not implemented: {bad}; implemented: {stages.IMPLEMENTED}", file=sys.stderr)
        return 2

    on_ac, pct = _power_state()
    if on_ac is False and not a.allow_battery:
        print(f"refusing to start on battery power ({pct}%); plug in or pass --allow-battery", file=sys.stderr)
        return 3
    _keep_awake()
    cfg = load_config(a.config)
    ctx = RunContext(cfg, a.run_id, todo, resume=a.resume)
    ctx.log.info("power: on AC = %s, battery %s%%", on_ac, pct)
    ctx.log.info("run %s | stages %s | commit %s dirty=%s", ctx.run_id, todo, ctx.manifest["git"]["commit"][:10],
                 ctx.manifest["git"]["dirty"])
    if any(s in stages.R_STAGES for s in todo):
        ctx.manifest["r"] = r_versions(cfg)
        ctx._write_manifest()
        ctx.log.info("R: %s", ctx.manifest["r"])
    rc = 0
    try:
        for s in todo:
            ctx.run_stage(s, stages.get(s))
    except Exception:
        rc = 1
    finally:
        ctx.close()
    ctx.log.info("run finished with exit code %d; verify with: python run_all.py --verify %s", rc, ctx.run_dir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
