# AGENTS.md — How to work in this repo

## Goals
- Prefer correctness, clarity, and tests over cleverness.
- Keep changes narrowly scoped and consistent with the existing style.
- Add or update tests for any behavior change; justify when tests are skipped.

## Workflow
- If running on cloud use python 3.13 and install in editable mode before running tools: `pip install -e .`. Additionally "git clone https://github.com/OHBA-analysis/osl-ephys" and install in editable mode also.
- If running locally use ephys-gpt environment, no need for any installs.
- Consult `README.md` and docs under `docs/` (`architecture.md`, `models.md`, `modal_commands.md`, `dev_commands.md`, `testing_strategy.md`) for repo layout, commands, and config structure.

## Must-run checks (report commands + results)
- Tests: `pytest -q tests`
- Formatting: `black --diff .` to show changes, then `black .` if you need to reformat.

## Definition of done
- Explain the rationale or root cause behind the change.
- Add/adjust tests when behavior changes (or note why not applicable).
- All required checks above pass.
- Refactor related code when it improves clarity or correctness.
