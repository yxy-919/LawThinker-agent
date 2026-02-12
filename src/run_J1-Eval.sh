#!/bin/bash
set -e

python ./src/Eval/bench/KQ/KQ.py
python ./src/Eval/bench/LC/LC.py
python ./src/Eval/bench/CD/CD.py
python ./src/Eval/bench/DD/DD.py
python ./src/Eval/bench/CI/CI.py
python ./src/Eval/bench/CR/CR.py