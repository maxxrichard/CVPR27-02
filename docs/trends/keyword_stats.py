"""Reproducible title-keyword statistics for CVPR 2025 vs CVPR 2026.

Sources (community mirrors of the official accepted-paper lists; see docs/trends/README.md):
  * CVPR 2025: gisbi-kim/cvml-paper-phylogeny  (papers.json, venue == "CVPR", year == "2025")
  * CVPR 2026: gisbi-kim/cvpr2026-explorer      (data/cvpr2026_papers.json)
Only titles are used so that both years are treated identically (the 2025 list has no abstracts).
Usage: python keyword_stats.py <phylogeny_papers.json> <cvpr2026_papers.json>
"""
import json, re, sys, collections

phylo, cvpr26 = sys.argv[1], sys.argv[2]
P25 = [p['title'] for p in json.load(open(phylo)) if p.get('venue') == 'CVPR' and str(p.get('year')) == '2025']
P26 = [p['title'] for p in json.load(open(cvpr26))]
print(f'CVPR 2025 titles: {len(P25)}   CVPR 2026 titles: {len(P26)}')

AREAS = collections.OrderedDict([
 ('Vision-language / multimodal (VLM, MLLM, CLIP, vision-language)', r'vision[- ]language|\bVLM|\bMLLM|\bLVLM|\bCLIP\b|multimodal|multi-modal'),
 ('Generative: diffusion / flow matching', r'diffusion|flow[- ]matching|rectified flow'),
 ('Video generation / world models', r'video generation|video diffusion|world model|text-to-video|image-to-video'),
 ('3D reconstruction / novel view / Gaussian splatting / NeRF', r'3D reconstruction|gaussian splatting|3DGS|NeRF|radiance field|novel view|multi-view stereo|structure[- ]from[- ]motion|\bSfM\b|feed-forward 3D|VGGT'),
 ('Embodied AI / robotics / VLA', r'robot|embodied|manipulation|vision[- ]language[- ]action|\bVLA\b|navigation'),
 ('Open-vocabulary recognition / segmentation', r'open[- ]vocab'),
 ('Test-time adaptation / test-time training / transductive', r'test[- ]time|transducti'),
 ('Zero-shot / few-shot / training-free adaptation', r'zero[- ]shot|few[- ]shot|training[- ]free'),
 ('Segmentation / dense prediction', r'segment|depth estimation|optical flow'),
 ('Self-supervised / representation learning / foundation model features', r'self[- ]supervis|representation learning|foundation model|DINO'),
 ('Medical imaging', r'medical|clinical|pathology|histopath|radiolog|\bMRI\b|\bCT\b'),
 ('Robustness / OOD / domain generalization / adaptation', r'robust|out-of-distribution|\bOOD\b|domain (generalization|adaptation|shift)|corruption'),
 ('Efficiency / compression / distillation / token pruning', r'efficient|pruning|quantiz|distillation|token (reduction|merging|compression)|lightweight'),
 ('Model merging / task vectors', r'model merging|task vector|merging'),
 ('Calibration / uncertainty / conformal', r'calibrat|uncertainty|conformal'),
 ('Video understanding', r'video understanding|action recognition|temporal grounding|video question'),
 ('Agents', r'\bagent'),
 ('Test-time (scaling|compute|training) phrases', r'test[- ]time (scaling|compute|training)'),
])
print(f"{'area':75s} {'2025':>6s} {'2026':>6s} {'%25':>6s} {'%26':>6s} {'growth':>7s}")
for name, pat in AREAS.items():
    r = re.compile(pat, re.I)
    n25 = sum(1 for t in P25 if r.search(t)); n26 = sum(1 for t in P26 if r.search(t))
    g = (n26/len(P26)) / max(n25/len(P25), 1e-9)
    print(f'{name:75s} {n25:6d} {n26:6d} {100*n25/len(P25):6.1f} {100*n26/len(P26):6.1f} {g:7.2f}x')

print('\n--- CVPR 2025 titles matching test-time/transductive AND (CLIP|vision-language|zero-shot|VLM):')
for t in P25:
    if re.search(r'test[- ]time|transducti', t, re.I) and re.search(r'CLIP|vision[- ]language|zero[- ]shot|\bVLM', t, re.I):
        print(' *', t)
print('\n--- CVPR 2026 titles matching the same filter:')
for t in P26:
    if re.search(r'test[- ]time|transducti', t, re.I) and re.search(r'CLIP|vision[- ]language|zero[- ]shot|\bVLM', t, re.I):
        print(' *', t)
