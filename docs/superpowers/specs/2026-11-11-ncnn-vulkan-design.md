# PBI-095, rodada 4: YOLOv10s com ncnn e Vulkan

Data: 2026-11-11 · Card: PBI-095 (Todoist `6hxmXq88f3mwWRf6`) · Status: design aprovado · Rodadas anteriores: AOTInductor e TensorRT (GPU NVIDIA), baseline em CPU

## Objetivo

Medir o caminho de GPU **independente de fabricante**: a YOLOv10s convertida para ncnn e executada via Vulkan, a partir de C++, na RTX 3050 e na GPU integrada Intel, comparada com as rodadas anteriores sob o mesmo protocolo e a mesma validação.

## Decisões (com o Nicolas, 2026-11-11)

- Runtime: **ncnn 20261020** do repositório do CachyOS (compilado com Vulkan, INT8 e BF16), instalado no sistema. Conversão com **pnnx 20261020** (PyPI) no `.venv`.
- Dispositivos Vulkan (ambos com `shaderFloat16` e `storageBuffer16BitAccess`):
  - **RTX 3050 6GB Laptop** (driver NVIDIA 620.70): tem `VK_KHR_cooperative_matrix` e `VK_NV_cooperative_matrix2`, ou seja, acesso aos tensor cores pelo Vulkan;
  - **Intel Graphics (RPL-S)**, integrada (Mesa): sem cooperative matrix. O Nicolas pediu explicitamente o benchmark nela.
- Variantes: ncnn Vulkan na NVIDIA (FP32 e FP16), ncnn Vulkan na Intel (FP32 e FP16) e, como extra, ncnn na CPU (FP32).
- **TopK fora do grafo, se preciso**: se o ncnn não suportar a seleção top-k do head end2end, o grafo do ncnn termina no head one-to-one bruto e a seleção dos 300 melhores roda em C++. Esse tempo é medido separadamente e **somado** ao total, para comparação justa com os backends que fazem isso dentro do grafo. A saída final continua `[1, 300, 6]`.

## Componentes

1. `ncnn/export_ncnn.py`: converte a YOLOv10s (mesmo grafo e entrada estática 1×3×640×640 das rodadas anteriores) com pnnx para `artifacts/ncnn/yolov10s.ncnn.{param,bin}`, documentando o que ficou dentro e fora do grafo.
2. `ncnn/cpp/` (CMake, `find_package(ncnn)`), binário `yolo_ncnn` com os modos `validate`, `bench` e `detect`, usando `native/common.h`, com `--device nvidia|intel|cpu` e `--precision fp32|fp16`.
3. `native/bench_ncnn.sh` e o relatório `docs/ncnn-vulkan-results.md`.

## Validação

- Mesmas referências (`artifacts/ref/`), mesmas 8 entradas e mesmas tolerâncias das rodadas anteriores: FP32 contra a referência FP32 estrita (caixas até 0,5 px, score até 0,001), FP16 contra a referência FP16 (2 px, 0,005), detecções com score ≥ 0,25 com mesma quantidade, mesma classe e IoU ≥ 0,95, diferenças medidas em pares casados por IoU.
- `detect --check` em `bus.jpg` com o letterbox em C++ (IoU ≥ 0,90).
- Nenhuma tolerância é afrouxada; uma falha é reportada com números e investigada.

## Benchmark

- Protocolo das rodadas de GPU: 50 iterações de aquecimento e 500 medidas, cada uma sincronizada. Na CPU, o protocolo da rodada 3 (20 + 200, com 6 threads nos P-cores e resfriamento).
- Duas medições por variante Vulkan: **entrada já na GPU** (VkMat enviado uma vez; comparável ao TensorRT) e **da memória do host ao host** (inclui upload da entrada e download da saída).
- Resfriamento entre variantes e 3 rodadas com ordem trocada, reportando a mediana, como na rodada 3.
- Métricas: latência média, p50 e p95, FPS, tempo do top-k em C++ (se houver), memória de GPU do processo (NVML, só na NVIDIA), pico de RSS, tempo de carga e da primeira inferência (inclui compilação de shaders/pipeline cache), tamanho do modelo.
- Configurações do ncnn impressas em cada bloco (fp16 storage/arithmetic, cooperative matrix, packing, threads).

## Riscos conhecidos

- Operações do head (TopK, GatherElements) e da atenção (PSA) sem suporte no ncnn.
- A primeira execução no Vulkan compila pipelines; a carga a frio pode ser longa.
- A GPU integrada divide memória e consumo de energia com a CPU; a temperatura afeta as duas.
