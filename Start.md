# Project Setup and Run Guide

## 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <repo-name>
```

## 2. Create and Activate Virtual Environment
```
python -m venv .env
```

Activate the virtual environment
```
On Windows:
.env\Scripts\activate

On Linux/Mac:
source .env/bin/activate
```

## 3. Install Dependencies
For GPU enabled pytorch(cuda version-12.6):

Get your version compatible pytorch at: [Pytorch](https://pytorch.org/get-started/locally/)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
For CPU enabled pytorch and other dependencies:
```
pip install -r requirements.txt
```

## 4. Run the Program
```
python main.py
```

