"""
Baixa os pesos pré-treinados do CSRNet via Hugging Face.
Execute uma vez antes de rodar o app:
    python download_weights.py
"""
import subprocess
import sys
from pathlib import Path

WEIGHTS_DIR  = Path("weights")
WEIGHTS_PATH = WEIGHTS_DIR / "csrnet_sha.pth"
HF_REPO      = "muasifk/CSRNet"
HF_FILENAME  = "CSRNet.pth"


def download():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    if WEIGHTS_PATH.exists():
        print(f"Pesos já existem em {WEIGHTS_PATH}")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Instalando huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        from huggingface_hub import hf_hub_download

    print(f"Baixando {HF_FILENAME} de {HF_REPO} ...")
    cached = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
    import shutil
    shutil.copy(cached, WEIGHTS_PATH)
    print(f"Salvo em {WEIGHTS_PATH}")


if __name__ == "__main__":
    download()
