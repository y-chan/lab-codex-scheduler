"""Deterministic Cartesian sweep expansion from a bounded JSON specification."""

from __future__ import annotations

import itertools
import json
import math
import re

from lab_codex_scheduler.config import Environment


MAX_SWEEP_JOBS = 100
_PARAMETER = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")
_SHELLS = {"sh", "bash", "zsh", "fish", "cmd", "powershell", "pwsh"}


def expand_sweep(environment: Environment, specification: str | bytes) -> list[tuple[str, ...]]:
    """Expand ``command_argv`` with parameter lists, preserving argv boundaries."""
    try:
        payload = json.loads(specification)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("sweep specification must be valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"command_argv", "parameters"}:
        raise ValueError("sweep specification needs exactly command_argv and parameters")
    template = payload["command_argv"]
    parameters = payload["parameters"]
    if not isinstance(template, list) or not 1 <= len(template) <= 128 \
            or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in template):
        raise ValueError("command_argv must be a nonempty array of safe strings")
    if template[0].rsplit("/", 1)[-1].lower() in _SHELLS:
        raise ValueError("shell commands are not accepted in sweep command_argv")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("parameters must be a nonempty object of value arrays")
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
           or not isinstance(values, list) or not values for name, values in parameters.items()):
        raise ValueError("parameter names must be valid identifiers with nonempty arrays")
    names = sorted(parameters)
    combinations = math.prod(len(parameters[name]) for name in names)
    if combinations > MAX_SWEEP_JOBS:
        raise ValueError(f"sweep expands to {combinations} jobs; maximum is {MAX_SWEEP_JOBS}")
    for name in names:
        for value in parameters[name]:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"parameter {name} contains a non-finite number")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"parameter {name} values must be strings, numbers, or booleans")
            if isinstance(value, str) and "\x00" in value:
                raise ValueError(f"parameter {name} contains NUL")
    used = {match for arg in template for match in _PARAMETER.findall(arg)}
    if used != set(names):
        raise ValueError("every parameter must appear in command_argv and placeholders must be known")
    # Reject unmatched braces or format expressions rather than interpreting them.
    for arg in template:
        residual = _PARAMETER.sub("", arg)
        if "{" in residual or "}" in residual:
            raise ValueError("command placeholders must use the exact {name} form")
    expanded: list[tuple[str, ...]] = []
    for values in itertools.product(*(parameters[name] for name in names)):
        mapping = {name: _stringify(value) for name, value in zip(names, values, strict=True)}
        argv = tuple(_PARAMETER.sub(lambda match: mapping[match.group(1)], arg) for arg in template)
        # Environment command construction performs execution-mode-specific validation too.
        environment.command(list(argv))
        expanded.append(argv)
    return expanded


def _stringify(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
