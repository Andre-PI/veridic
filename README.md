# Veridic — Estimativa de Público

Estimativa de público em eventos com **CSRNet** (density map) + **método de Jacobs** (área × densidade), usando footage de drone DJI.

## Instalação rápida

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

O script instala dependências, baixa os pesos do CSRNet e inicia o app.

## Uso no dia do evento

1. **Drone:** DJI Mini 4 Pro, altitude 50m, gimbal -90° (câmera para baixo)
2. **Gravação:** vídeo lento cobrindo o venue inteiro no momento de maior público
3. **No app:**
   - Preset: `Show / Festa junina`
   - Altitude: `50`, FOV: `82.1`
   - Área do venue (m²) se souber
   - Selecione a densidade observada
   - Faça upload e clique em Analisar

## Métodos

| Método | Como funciona | Melhor para |
|---|---|---|
| **CSRNet** | Rede neural que gera mapa de densidade | Multidões médias (50–500 pessoas), vista aérea vertical |
| **Jacobs** | Área detectada × fator de densidade | Grandes eventos (500+), especialmente com área conhecida |

A **concordância** entre os dois indica confiança no resultado. Verde ≥ 80%, amarelo 50–80%, vermelho < 50%.

## Requisitos

- Python 3.9+
- ffmpeg (opcional, para leitura de metadados do drone)
- GPU CUDA (opcional, acelera o processamento)

## Configuração da câmera

| Drone | FOV horizontal |
|---|---|
| DJI Mini 4 Pro | 82.1° |
| DJI Mini 3 Pro | 82.1° |
| DJI Air 3 | 82° |
| DJI Mavic 3 | 84° |
