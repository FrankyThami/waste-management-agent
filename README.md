# Waste Management Agent

Final-year project: a software agent for robotic waste sorting. It uses a
vision-language model (Gemini) to look at a camera scene, understand what each
object is, and decide what the robot should do, using a small set of tools.

## What it does

- Looks at the scene with an Intel RealSense camera (RGB, optional depth).
- Classifies each waste object: paper, metal, organic, unsorted, plastic, glass.
- Chooses and runs actions through tools: observe, pick, place, stop.
- Checks every action with safety guardrails and logs every decision.
- Robot hardware calls go through a simulated hardware layer (HAL) for now.

## Project goal (in progress)

Give the agent a plain-English command, for example "plastic goes to the red
bin, paper to the blue bin, do plastic first". The agent understands the rules,
finds the objects and bins in the live scene, and adapts when things move.

## Folder layout

```
src/
  policy.py, schemas.py, utils.py   shared settings, data shapes, helpers
  reasoning/     agent instructions (prompts.py)
  actions/       tools, LangChain tool wrappers, simulated hardware layer (hal.py)
  context/       task state (state.py)
  models/        Gemini connection (gemini_classifier.py) and image prompts (prompts_v1.py)
  interface/     terminal entry point (run_agent.py)
  reliability/   guardrails, middleware, task logger
tests/           unit tests and demos
config/          bin positions and camera calibration
logs/            run logs (agent_log.csv, session_summary.csv)
scripts/         helper scripts (connection test, camera check)
```

Earlier experiments still sit in the project root and will be moved later:
`adaptive_validation/`, `offline_reasoning/`, `check_labels.py`,
`run_accuracy_test.py`.

## Setup (Windows PowerShell)

```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then open `.env` and put your real Gemini API key in it.

## Run

Run every command from the project root, with the virtual environment active.

Start the agent (the observe tool reads the RealSense camera, so plug it in first):

```
python -m src.interface.run_agent
```

Check the Gemini connection:

```
python scripts/test_connection.py
```

Run all tests (every line should say exit code 0):

```
Get-ChildItem tests\test_*.py | ForEach-Object { python -m "tests.$($_.BaseName)" *> $null; "$($_.Name): exit code $LASTEXITCODE" }
```

## Status

Basic agent works end to end with a simulated robot. The upgrade to
plain-English commands and moving scenes is in progress.