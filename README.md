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

## Sobre o modelo CSRNet

O modelo usado é o **CSRNet treinado no ShanghaiTech Part A (SHA)** — o mais acessível e amplamente testado para contagem de multidões.

**Limitação importante:** o SHA foi treinado em câmeras de nível de rua. Vídeos de drone a 50m em vista nadir têm uma perspectiva diferente do conjunto de treinamento, o que faz o CSRNet **subcontar em multidões grandes e densas** (típico de shows com milhares de pessoas).

Por isso, neste sistema o CSRNet tem papel de **apoio visual** (heatmap e contornos), enquanto o **método de Jacobs com densidade informada pelo observador é o estimador principal** para grandes eventos.

**Modelos mais precisos existem**, mas com trade-offs:

| Modelo | Vantagem | Desvantagem |
|---|---|---|
| CSRNet SHA (atual) | Fácil de baixar, bem documentado | Treinado em nível de rua |
| DM-Count (QNRF) | Mais preciso, dataset mais diverso | Requer troca de arquitetura |
| Modelo treinado no evento | Calibrado para o venue/câmera exatos | Requer coleta e labeling de dados |

Para uso profissional contínuo, o caminho ideal é **fine-tuning** após os primeiros eventos: usar as filmagens reais com a contagem do Jacobs como referência para treinar um modelo específico para o venue e câmera utilizados.

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
