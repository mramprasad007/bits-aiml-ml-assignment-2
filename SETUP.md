# Setup Instructions

```bash
conda create -n bits python=3.11
conda activate bits
pip install -r requirements.txt
python train_models.py
streamlit run app.py