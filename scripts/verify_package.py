from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    artifact_root = ROOT / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="glassbox-package-", dir=artifact_root
    ) as temporary:
        work = Path(temporary)
        dist = work / "dist"
        installed = work / "installed"
        artifacts = work / "artifacts"

        run(
            [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist)],
            cwd=ROOT,
        )
        distributions = sorted(dist.iterdir())
        run([sys.executable, "-m", "twine", "check", *map(str, distributions)], cwd=ROOT)

        wheels = list(dist.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one wheel, found {len(wheels)}")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "--target",
                str(installed),
                str(wheels[0]),
            ],
            cwd=work,
        )

        smoke_code = (
            "import json,sys;"
            f"sys.path.insert(0,{str(installed)!r});"
            "import glassbox_audit;"
            "from glassbox_audit.package_smoke import run_installed_smoke;"
            f"result=run_installed_smoke({str(artifacts)!r});"
            "result['package_file']=glassbox_audit.__file__;"
            "print(json.dumps(result))"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", smoke_code],
            cwd=work,
            env={**os.environ, "PYTHONPATH": ""},
            text=True,
            capture_output=True,
            check=True,
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        if not Path(result["package_file"]).is_relative_to(installed):
            raise RuntimeError("Smoke imported the source tree instead of the built wheel")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
