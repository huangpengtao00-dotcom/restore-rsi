"""What the record relay will and will not execute.

The relay reads `restore` commands out of free-form model text, which puts it
between two failure modes that both actually happened:

- anchor to the line start and nothing matches, because the served model wraps
  every command in its own tool-call markup -> the first full run scored 0 on
  all three tasks and looked like a strategy problem;
- match the bare phrase anywhere and the model's prose gets executed -> "I
  should NOT restore run sharpen_unsharp" ran sharpen_unsharp (2026-09-03
  audit).

So the cases below are the contract, not examples. Every "should match" string
is a real reply shape seen in work/run.log or a standard one (markdown fence,
shell prompt); every "should not" is a mention rather than a command.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REEF_EXAMPLE = Path(__file__).resolve().parent.parent / "reef_example"


@pytest.fixture(scope="module")
def command_pattern():
    """reef_example/run.py's COMMAND, imported without running the loop."""
    if str(REEF_EXAMPLE) not in sys.path:
        sys.path.insert(0, str(REEF_EXAMPLE))
    spec = importlib.util.spec_from_file_location("restore_relay_run", REEF_EXAMPLE / "run.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # reef-client only installed with the `reef` group
        pytest.skip(f"reef_example deps unavailable: {exc}")
    return module.COMMAND


# The served model's own tool-call markup, verbatim from work/run.log.
DSML_REPLY = (
    '<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name="shell">\n'
    '<｜｜DSML｜｜parameter name="command" string="true">cd /repo && '
    "restore diagnose /repo/tasks/data/haze_low_light_00/input_000.png"
    "</｜｜DSML｜｜parameter>\n</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜tool_calls>"
)

COMMANDS = [
    (DSML_REPLY, [("diagnose", "/repo/tasks/data/haze_low_light_00/input_000.png")]),
    ("restore run lowlight_gamma in.png out1.png", [("run", "lowlight_gamma in.png out1.png")]),
    ("```bash\nrestore run denoise_bilateral in.png out1.png\n```", [("run", "denoise_bilateral in.png out1.png")]),
    ("$ restore diagnose in.png", [("diagnose", "in.png")]),
    ("First I will `restore diagnose in.png` to read the numbers.", [("diagnose", "in.png")]),
    (
        "restore run lowlight_gamma a.png b.png\nrestore run dehaze_dcp b.png c.png",
        [("run", "lowlight_gamma a.png b.png"), ("run", "dehaze_dcp b.png c.png")],
    ),
]

MENTIONS = [
    "I should NOT restore run sharpen_unsharp because there is no blur.",
    "If I were to restore run dehaze_dcp first, the image would get darker.",
    "My mistake was to restore run sharpen_unsharp twice.",
    "The next thing to do is restore run lowlight_gamma on it.",
    "The final image is /tmp/x/step2_dehaze_dcp.png",
]


@pytest.mark.parametrize("reply,expected", COMMANDS)
def test_commands_in_command_position_are_executed(command_pattern, reply, expected):
    assert command_pattern.findall(reply) == expected


@pytest.mark.parametrize("reply", MENTIONS)
def test_mentions_in_prose_are_not_executed(command_pattern, reply):
    assert command_pattern.findall(reply) == []
