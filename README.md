# YOLOv10s na webcam

Experimento de inferência com YOLOv10-small em uma GPU NVIDIA, com imagem anotada, FPS e uso de memória. É necessário Linux, uma webcam V4L2 (ou arquivo de vídeo), driver NVIDIA compatível com CUDA 12.8 e [uv](https://docs.astral.sh/uv/). O projeto usa Python 3.12; o uv pode instalá-lo automaticamente.

## Instalação

```bash
uv sync
```

Na primeira execução, o Ultralytics baixa automaticamente os pesos `yolov10s.pt`. Eles não são versionados.

## Uso

```bash
uv run webcam_yolo.py
uv run webcam_yolo.py --half
uv run webcam_yolo.py --no-show --frames 300
```

A janela exibe FPS suavizado, tempo do modelo e memória; pressione **q** ou **Esc**, ou feche a janela para sair. Ctrl+C também encerra e imprime o resumo. O programa exige CUDA no dispositivo 0 e não usa CPU como alternativa.

Use `--camera 1` para outra câmera, `--width` e `--height` para solicitar outra resolução, `--imgsz` para alterar o tamanho de inferência e `--conf` para o limiar de confiança. A câmera solicita MJPG a 30 FPS; a resolução e o FPS efetivos são impressos. Para testar com um vídeo, use `--source caminho/do/video.mp4`. `--no-show` dispensa a janela, e `--frames N` limita a medição a N quadros (0 significa sem limite). Cinco quadros de aquecimento são excluídos das estatísticas.

O FPS do loop inclui captura, inferência, desenho e apresentação (quando habilitada). A latência do modelo mede a chamada `predict`, sincronizada com a GPU, e é apresentada como média, p50 e p95 em milissegundos.

## Memória da GPU

Todos os valores são em MiB:

- **torch allocated**: memória ocupada por tensores do PyTorch; o pico é o máximo desde o fim do aquecimento.
- **torch reserved**: memória reservada pelo alocador do PyTorch, incluindo memória em cache reutilizável; o pico também é medido após o aquecimento.
- **process GPU (NVML)**: memória atribuída pelo driver a este processo de computação, incluindo memória fora do alocador PyTorch.
- **GPU total used / total**: memória usada por todos os processos e capacidade total da GPU, segundo o driver.

As métricas do driver usam NVML e aparecem como `n/a` quando indisponíveis ou quando o processo não é encontrado. Elas não precisam coincidir com as métricas do PyTorch.
