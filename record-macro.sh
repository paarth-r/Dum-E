#!/usr/bin/env zsh
# Record a hand-guided macro: prompts for a name and a digit keybind, counts down 3-2-1,
# cuts torque so you can move the arm by hand, and records until you hit space. Play it
# back in `dume run` by pressing that digit.
cd "$(dirname "$0")" && exec .venv/bin/dume record "$@"
