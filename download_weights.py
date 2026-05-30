"""
Baixa os pesos pré-treinados do CSRNet (ShanghaiTech Part A).
Execute uma vez antes de rodar o app:
    python download_weights.py
"""
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path("weights")
WEIGHTS_PATH = WEIGHTS_DIR / "csrnet_sha.pth"

# Mirror público com pesos do CSRNet treinado no ShanghaiTech Part A
URL = "https://github.com/leeyeehoo/CSRNet-pytorch/releases/download/v1.0/best_model_A.pth"

def download():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    if WEIGHTS_PATH.exists():
        print(f"Pesos já existem em {WEIGHTS_PATH}")
        return
    print(f"Baixando pesos de {URL} ...")
    urllib.request.urlretrieve(URL, WEIGHTS_PATH)
    print(f"Salvo em {WEIGHTS_PATH}")

if __name__ == "__main__":
    download()
