#!/usr/bin/env bash
# setup.sh — Veridic: setup automático (macOS / Linux)
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }

echo ""
echo "  Veridic — setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Python 3.9+
if ! command -v python3 &>/dev/null; then
    fail "Python 3 não encontrado. Instale em https://python.org"
fi
PY_VER=$(python3 -c "import sys; print(sys.version_info.minor)")
[ "$PY_VER" -ge 9 ] || fail "Python 3.9+ necessário (encontrado 3.$PY_VER)"
ok "Python 3.$(python3 -c 'import sys; print(sys.version_info.minor)')"

# ffmpeg (metadados do drone)
if command -v ffprobe &>/dev/null; then
    ok "ffprobe encontrado"
else
    warn "ffprobe não encontrado — metadados do drone não serão lidos automaticamente"
    if command -v brew &>/dev/null; then
        echo "  Instalar via Homebrew? (s/n)"
        read -r resp
        [[ "$resp" == "s" ]] && brew install ffmpeg && ok "ffmpeg instalado"
    else
        warn "Instale manualmente: https://ffmpeg.org/download.html"
    fi
fi

# venv
if [ ! -d "venv" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
ok "Ambiente virtual ativo"

# dependências
echo "Instalando dependências..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
ok "Dependências instaladas"

# pesos do modelo
if [ ! -f "weights/csrnet_sha.pth" ]; then
    echo "Baixando pesos do CSRNet..."
    python download_weights.py
else
    ok "Pesos do modelo encontrados"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "Setup concluído!"
echo ""
echo "Para iniciar o Veridic:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""

# pergunta se já quer rodar
echo "Iniciar agora? (s/n)"
read -r resp
if [[ "$resp" == "s" ]]; then
    streamlit run app.py
fi
