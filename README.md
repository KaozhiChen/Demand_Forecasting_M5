# M5 Demand Forecasting Project

This project implements and compares three deep learning models for time-series demand forecasting: **LSTM**, **Transformer**, and **Transformer-D**.

## 1. Environment Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Project Structure

```
Demand_Forecasting_M5/
├── Data/
│   ├── Row_Data/              # Raw M5 dataset files (calendar.csv, sales_train_evaluation.csv)
│   ├── training_data.csv.zip  # Compressed processed data (extract before use)
│   └── data_analysis.ipynb    # Data preprocessing notebook
├── models/
│   ├── lstm.py
│   ├── transformer.py
│   └── transformer_d.py
├── models_and_metrics/        # Saved models and predictions
├── utils/
│   └── result_analyze.ipynb   # Results visualization
├── config.py                  # Configuration settings
├── data.py                    # Data loading
├── train.py                   # Training script
└── eval.py                    # Evaluation script
```

## 3. Prepare Training Data

Extract the compressed training data:

```bash
cd Data
unzip training_data.csv.zip
```

This will create `training_data.csv` (~216MB) in the `Data/` directory.


## 4. Run Commands

### Train a Model

```bash
# Train LSTM
python train.py --model lstm --epochs 20

# Train Transformer
python train.py --model transformer --epochs 20

# Train Transformer-D
python train.py --model transformer_d --epochs 20
```

### Evaluate a Model

```bash
python eval.py --model lstm --checkpoint models_and_metrics/lstm_best.pth
python eval.py --model transformer --checkpoint models_and_metrics/transformer_best.pth
python eval.py --model transformer_d --checkpoint models_and_metrics/transformer_d_best.pth
```

### Analyze Results

Open `utils/result_analyze.ipynb` to visualize results and generate comparison plots.



**Configuration**: All hyperparameters are set in `config.py`. Training uses early stopping (patience=3) and MLflow for experiment tracking.
