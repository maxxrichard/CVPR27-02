#!/usr/bin/env bash
# Downloads the blueprint paper (StatA, CVPR 2025) into this folder.
set -e
cd "$(dirname "$0")"
curl -L -o StatA_CVPR2025.pdf "https://arxiv.org/pdf/2501.03729" || \
curl -L -o StatA_CVPR2025.pdf "https://openaccess.thecvf.com/content/CVPR2025/papers/Zanella_Realistic_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2025_paper.pdf"
curl -L -o TransCLIP_NeurIPS2024.pdf "https://arxiv.org/pdf/2406.01837" || true
curl -L -o MOON_ICML2026.pdf "https://arxiv.org/pdf/2607.15851" || true
ls -la *.pdf
