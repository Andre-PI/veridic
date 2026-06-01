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

--------#######---------

cd veridic
setup.bat
```

O setup cria o ambiente virtual, instala as dependências e baixa os pesos do CSRNet automaticamente via Hugging Face.

## Opcional( Instalar UV gerenciador ultra rapido )

```bash
brew install uv (macOS usando homebrew)
```

## Criar um abiente virtual

```bash
python3 -m venv .venv

ou

uv venv (usando uv)

```

## Iniciar Ambiente Virtual

```bash
source .venv/bin/activate
```

## Instalar as dependencias usando o file `requirements.txt`

```bash
uv pip install -r requirements.txt (usando uv)

ou

pip install -r requirements.txt
```

## Instalar weights para CSRNet

```bash
uv run python3 download_weights.py
```

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
