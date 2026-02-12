#!/bin/bash
set -e

python ./src/run.py \
    --scenario J1Bench.Scenario.KQ \
    --general_public Agent.General_public.ConsultQwen3_32B \
    --trainee Agent.Trainee.ConsultQwen3_32B \
    --save_path "./src/data/dialog_history/32b/KQ_dialog_history.jsonl" \

python ./src/run.py \
    --scenario J1Bench.Scenario.LC \
    --general_public Agent.General_public.LC_Qwen3_32B \
    --trainee Agent.Trainee.LC_Qwen3_32B \
    --save_path "./src/data/dialog_history/32b/LC_dialog_history.jsonl" \

python ./src/run.py \
    --scenario J1Bench.Scenario.CD \
    --specific_character Agent.Specific_character.Qwen332B_CD \
    --lawyer Agent.Lawyer.Qwen3_32B_CD \
    --save_path "./src/data/dialog_history/32b/CD_dialog_history.jsonl" \

python ./src/run.py \
    --scenario J1Bench.Scenario.DD \
    --specific_character Agent.Specific_character.Qwen332B_DD \
    --lawyer Agent.Lawyer.Qwen3_32B_DD \
    --save_path "./src/data/dialog_history/32b/DD_dialog_history.jsonl" \

python ./src/run.py \
    --scenario J1Bench.Scenario.CI \
    --plaintiff Agent.Plaintiff.Qwen3_32B_CI \
    --defendant Agent.Defendant.Qwen3_32B_CI \
    --judge Agent.Judge.Qwen3_32B_CI \
    --save_path "./src/data/dialog_history/32b/CI_dialog_history.jsonl" \

python ./src/run.py \
    --scenario J1Bench.Scenario.CR \
    --defendant Agent.Defendant.Qwen3_32B_CR \
    --lawyer Agent.Lawyer.Qwen3_32B_CR \
    --procurator Agent.Procurator.Qwen3_32B_CR \
    --judge Agent.Judge.Qwen3_32B_CR \
    --save_path "./src/data/dialog_history/32b/CR_dialog_history.jsonl" \
