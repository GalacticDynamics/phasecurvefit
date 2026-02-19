#!/usr/bin/env -S uv run --script  # noqa: EXE001
# /// script
#    dependencies = ["nox", "nox_uv"]
# ///
"""Nox setup."""

import argparse
import shutil
from pathlib import Path

import nox
from nox_uv import session

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

DIR = Path(__file__).parent.resolve()


# =============================================================================
# Comprehensive sessions


@session(
    uv_groups=["lint", "test", "docs"],
    reuse_venv=True,
    default=True,
)
def all(s: nox.Session, /) -> None:
    """Run all checks (lint, test, docs)."""
    s.notify("lint")
    s.notify("test")
    s.notify("docs")


# =============================================================================
# Linting


@session(uv_groups=["lint"], reuse_venv=True)
def lint(s: nox.Session, /) -> None:
    """Run all linting checks."""
    s.notify("precommit")
    s.notify("pylint")


@session(uv_groups=["lint"], reuse_venv=True)
def precommit(s: nox.Session, /) -> None:
    """Run pre-commit hooks."""
    s.run("pre-commit", "run", "--all-files", *s.posargs)


@session(uv_groups=["lint"], reuse_venv=True)
def pylint(s: nox.Session, /) -> None:
    """Run PyLint."""
    s.run("pylint", "src/localflowwalk", *s.posargs)


# =============================================================================
# Testing


@session(uv_groups=["test"], reuse_venv=True)
def test(s: nox.Session, /) -> None:
    """Run the test suite."""
    s.notify("pytest", posargs=s.posargs)


@session(uv_groups=["test"], reuse_venv=True)
def pytest(s: nox.Session, /) -> None:
    """Run pytest with common options."""
    package_paths = [
        "README.md",
        "docs",
        "src/",
        "tests/",
    ]
    s.run("pytest", *package_paths, *s.posargs)


@session(uv_groups=["test"], reuse_venv=True)
def pytest_benchmark(s: nox.Session, /) -> None:
    """Run benchmark tests with pytest-benchmark."""
    s.run("pytest", "tests/benchmarks/", "--benchmark-only", "-v", *s.posargs)


@session(uv_groups=["test"])
def codspeed(s: nox.Session, /) -> None:
    """Run CodSpeed benchmarks for CI."""
    s.run("pytest", "tests/benchmarks/", "--codspeed", *s.posargs)


# =============================================================================
# Documentation


@session(uv_groups=["docs"], reuse_venv=True)
def docs(s: nox.Session, /) -> None:
    """Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    parser.add_argument("--output-dir", dest="output_dir", default="_build")
    args, posargs = parser.parse_known_args(s.posargs)

    if args.builder != "html" and args.serve:
        s.error("Must not specify non-HTML builder with --serve")

    s.chdir("docs")

    if args.builder == "linkcheck":
        s.run("sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs)
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        f"-d={args.output_dir}/doctrees",
        "-D",
        "language=en",
        ".",
        f"{args.output_dir}/{args.builder}",
        *posargs,
    )

    if args.serve:
        s.run("sphinx-autobuild", *shared_args)
    else:
        s.run("sphinx-build", "--keep-going", *shared_args)


# =============================================================================
# Packaging


@session(uv_groups=["dev"])
def build(s: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    s.run("python", "-m", "build")


# =============================================================================

if __name__ == "__main__":
    nox.main()
