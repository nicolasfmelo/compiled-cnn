# PBI-095, rodada 2: YOLOv10s com TensorRT na GPU

Data: 2026-11-11 · Card: PBI-095 (Todoist `6hxmXq88f3mwWRf6`) · Status: design aprovado · Rodada anterior: `2026-11-11-aoti-gpu-design.md`

## Objetivo

Levar a mesma YOLOv10s da rodada 1 para um engine TensorRT, executá-lo a partir de C++ **sem libtorch** e comparar com o eager e com o AOTInductor nas mesmas condições.

## Decisões (com o Nicolas, 2026-11-11)

- Caminho: **ONNX → engine TensorRT (`trtexec`) → runner C++ com a API do TensorRT (`nvinfer`)**. Torch-TensorRT fica para uma rodada separada, se fizer sentido.
- TensorRT **11.4.0.106 do AUR** (`tensorrt`), instalado no sistema para o **CUDA 13.4** do `/opt/cuda`, com cuDNN 9.28. O runner TensorRT usa o CUDA 13.4 do sistema; o runner AOTI continua com o CUDA 12.8 das wheels do torch. São binários separados.
- FP16 vem de um **ONNX exportado em half** e de um engine *strongly typed* (`--stronglyTyped`), não do flag de precisão do builder, que o TensorRT 10+ tornou obsoleto.
- O código comum aos runners (leitura das referências, comparação, letterbox, NVML, percentis) sai de `aoti/cpp/main.cpp` para `native/common.h`, sem dependência do torch, e passa a ser usado pelos dois.

## Componentes

1. `trt/export_onnx.py`: reutiliza o `load_model` de `aoti/export.py` (mesmo grafo end2end `[1, 300, 6]`, entrada estática `1×3×640×640`) e exporta `artifacts/yolov10s_fp32.onnx` e `artifacts/yolov10s_fp16.onnx`, verificados com `onnx.checker`.
2. `trt/build_engines.sh`: `trtexec --onnx … --saveEngine artifacts/yolov10s_{fp32,fp16}.plan --stronglyTyped` com as opções padrão (otimização nível 3), registrando o tempo de build e o log.
3. `trt/cpp/` (CMake), binário `yolo_trt` com os modos `validate`, `bench` e `detect`, iguais aos do `yolo_aoti` em argumentos e formato de saída.
4. `native/common.h` e `native/third_party/stb_image.h`: o código compartilhado e o `stb_image` movido de `aoti/cpp/third_party/`.
5. `native/bench_all.sh`: roda as 6 variantes alternando backends, na mesma sessão.

## Validação

- Mesmas referências (`artifacts/ref/`), mesmas 8 entradas e mesmas tolerâncias da rodada 1 (FP32: caixas ≤ 0,5 px, scores ≤ 1e-3; FP16: caixas ≤ 2 px, scores ≤ 5e-3; detecções com score ≥ 0,25 com mesma quantidade, mesma classe e IoU ≥ 0,95). `detect --check` em `bus.jpg` com IoU ≥ 0,90.
- Refinamento decidido na spec: as diferenças de caixa e de score passam a ser medidas **sobre os pares casados por IoU** (detecções com score ≥ 0,25), e não pela posição da linha. O TopK pode ordenar de forma diferente linhas com scores quase iguais, e isso não é erro numérico. A diferença linha a linha continua impressa como informativa. Vale para os dois runners, porque o código é comum.

## Benchmark

- Variantes, todas sem autotune do cuDNN: `eager-py`, `aoti-cpp` e `trt-cpp`, cada uma em FP32 e FP16.
- Protocolo da rodada 1: só a inferência, com a entrada já na GPU, 50 iterações de aquecimento e 500 medidas, cada uma sincronizada. No TensorRT, `enqueueV3` em um stream e `cudaStreamSynchronize`.
- Métricas da rodada 1 e mais, no TensorRT: tempo de build do engine, tamanho do `.plan` e tempo de desserialização.
- Configurações impressas em cada bloco; no TensorRT, informar se TF32 está habilitado no engine.

## Relatório

`docs/tensorrt-results.md`, em pt-BR: versões, comandos, `ldd` do `yolo_trt` (sem libtorch), operadores e avisos do parser ONNX, validação, a tabela comparativa dos 3 backends × 2 precisões e uma conclusão.

## Riscos conhecidos

- APIs do TensorRT 11 (redes *strongly typed*, `enqueueV3`, endereços de tensores por nome) diferentes de exemplos antigos.
- TopK do head do YOLOv10 e operações em FP16 no parser ONNX.
- O engine `.plan` é específico da GPU e da versão do TensorRT; não é portátil.
