# Veridic

Estimativa de público por imagens aéreas de drone. Duas abas:

- **CSRNet + Jacobs estimado** — processamento automático de foto ou vídeo via rede neural
- **Jacobs real (campo)** — contagem manual por grade com sorteio aleatório e código de auditoria

## Instalação

### macOS / Linux
```bash
git clone https://github.com/Andre-PI/veridic
cd veridic
chmod +x setup.sh
./setup.sh
```

### Windows
```bat
git clone https://github.com/Andre-PI/veridic
cd veridic
setup.bat
```

O setup cria o ambiente virtual, instala as dependências e baixa os pesos do CSRNet automaticamente via Hugging Face.

## Iniciar

```bash
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

streamlit run app.py
```

## Requisitos

- Python 3.9+
- ffmpeg / ffprobe — opcional, necessário para leitura automática de altitude e FOV de vídeos DJI
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: https://ffmpeg.org/download.html
