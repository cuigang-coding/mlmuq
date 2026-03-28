# MLMUQ

## Title
MLMUQ (PyTorch + Pyro implementation)

## Description
MLMUQ is a multimodal employability-prediction framework that combines meta-learning, Bayesian neural networks, and uncertainty-aware attention for calibrated prediction and few-shot adaptation.

## Dataset Information
This work uses three publicly accessible third-party employment outcome resources:

1. **Education-Career-Success (ECS)**
   - Source: https://www.kaggle.com/datasets/adilshamim8/education-and-career-success
   - Samples after preprocessing: 377

2. **Job-Placement (JP)**
   - Source: https://www.kaggle.com/datasets/mahad049/job-placement-dataset
   - Samples after preprocessing: 653

3. **Nigerian-Graduates (NG)**
   - Source: https://www.kaggle.com/code/obafemijoseph/nigerian-graduates
   - Samples after preprocessing: 4,873

Aggregated statistics:
- Total samples after preprocessing: 5,903
- Meta-training pool: 4,722
- Test pool: 1,181
- Shared prediction target: 5 employability outcome levels

## Data Preprocessing Steps
1. Map each dataset's native outcome fields to a shared 5-level employability target using deterministic rules.
2. Exclude records that do not have a usable target mapping.
3. Auto-classify feature columns into academic, skills, and experience modalities.
4. Mean-impute continuous variables.
5. Encode missing categorical values with a dedicated missing token before factorization.
6. Apply z-score normalization to continuous variables.
7. Use 80/20 train-test splitting and episodic sampling for meta-learning tasks.

## Code Information
The package includes:
- Data loading and preprocessing utilities
- Bayesian multimodal model definition
- Meta-learning training pipeline
- Command-line training entry point
- Python dependency file

## Usage Instructions
Run the following steps from the repository root.

### Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/mlmuq/requirements.txt
```

### Obtain the Third-Party Data
```bash
pip install kaggle
kaggle datasets download -d adilshamim8/education-and-career-success -p data/ecs --unzip
kaggle datasets download -d mahad049/job-placement-dataset -p data/jp --unzip
```

For NG, download or export the CSV from:
- https://www.kaggle.com/code/obafemijoseph/nigerian-graduates

Expected local paths:
- `data/ecs/education_career_success.csv`
- `data/jp/job_placement_dataset.csv`
- `data/ng/nigerian_graduates.csv`

### Train MLMUQ
```bash
PYTHONPATH=code python -m mlmuq.train_mlmuq \
  --ecs-csv data/ecs/education_career_success.csv \
  --jp-csv data/jp/job_placement_dataset.csv \
  --ng-csv data/ng/nigerian_graduates.csv \
  --outer-iters 2000 \
  --meta-batch-size 8 \
  --support-per-class 15 \
  --query-per-class 20 \
  --seed 42
```

### Output Files
Training writes results to `runs/mlmuq/` by default:
- `mlmuq_checkpoint.pt`
- `summary.json`

## Requirements
- Python 3.10+
- PyTorch 2.1.0
- Pyro 1.8.6
- pandas >= 2.0.0
- numpy >= 1.24.0

## Methodology
MLMUQ combines:
1. Meta-learning for rapid adaptation to new graduate cohorts.
2. Bayesian neural networks for epistemic and aleatoric uncertainty estimation.
3. Cross-modality attention for academic, skills, and experience feature fusion.

Default training configuration:
- `outer-iters=2000`
- `meta-batch-size=8`
- `support-per-class=15`
- `query-per-class=20`
- `seed=42`

## Citations
Dataset access references:

```bibtex
@misc{KaggleEducationCareerSuccess2026,
  author = {{adilshamim8}},
  title = {Education and Career Success},
  howpublished = {Kaggle dataset},
  year = {2025},
  url = {https://www.kaggle.com/datasets/adilshamim8/education-and-career-success},
  note = {Accessed 2026-03-28}
}

@misc{KaggleJobPlacementDataset2026,
  author = {{mahad049}},
  title = {Job Placement Dataset},
  howpublished = {Kaggle dataset},
  year = {2025},
  url = {https://www.kaggle.com/datasets/mahad049/job-placement-dataset},
  note = {Accessed 2026-03-28}
}

@misc{KaggleNigerianGraduatesNotebook2026,
  author = {{obafemijoseph}},
  title = {Nigerian Graduates},
  howpublished = {Kaggle notebook},
  year = {2018},
  url = {https://www.kaggle.com/code/obafemijoseph/nigerian-graduates},
  note = {Accessed 2026-03-28}
}
```

## License
This package is distributed under the MIT License. See `code/mlmuq/LICENSE` for the full text.
