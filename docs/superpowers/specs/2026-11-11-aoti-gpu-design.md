# PBI-095, rodada 1: YOLOv10s compilada com AOTInductor na GPU

Data: 2026-11-11 · Card: PBI-095 (Todoist `6hxmXq88f3mwWRf6`) · Status: design aprovado

## Objetivo

Compilar a YOLOv10s para um artefato nativo com AOTInductor (`.pt2`), carregá-lo e executá-lo a partir de C++ sem Python, validar as saídas contra o PyTorch eager e comparar desempenho e memória na GPU.

## Decisões (com o Nicolas, 2026-11-11)

- A primeira rodada é em **GPU** (RTX 3050 6GB Laptop, driver 615), não em CPU como sugeria o card. CPU, TensorRT, IREE, TVM e INT8 ficam para rodadas seguintes.
- Modelo: **YOLOv10s** (`yolov10s.pt` do Ultralytics), saída end2end `[1, 300, 6]` (x1, y1, x2, y2, score, classe) sem NMS, entrada estática `1×3×640×640`, batch 1.
- Compilador: **AOTInductor** com a API atual do torch 2.11: `torch.export.export` + `torch._inductor.aoti_compile_and_package`. `torch._export.aot_compile` está obsoleto e não é usado.
- Consumidor: **só C++** para validação e benchmark (`AOTIModelPackageLoader`). Python é usado apenas no build (exportar e compilar) e para gerar a referência eager.
- Sem o Astra neste experimento.

## Componentes

1. `aoti/export.py` (build-time):
   - carrega o modelo em eval com as camadas fundidas e expõe a saída end2end `[1, 300, 6]`;
   - exporta e compila `artifacts/yolov10s_fp32.pt2` e `artifacts/yolov10s_fp16.pt2`, registrando o tempo de compilação;
   - gera a referência em `artifacts/ref/`: tensores de entrada pré-processados (letterbox 640, pad 114, RGB, /255, CHW) e saídas do eager FP32 e FP16, em float32 bruto little-endian, com um manifesto em texto.
2. `aoti/bench_eager.py`: baseline eager FP32 e FP16 em Python, com o mesmo protocolo e o mesmo formato de saída do C++.
3. `aoti/cpp/` (CMake), binário `yolo_aoti`:
   - `validate`: roda o `.pt2` sobre as entradas de referência e imprime PASS/FAIL por imagem;
   - `bench`: latência, throughput e memória;
   - `detect <imagem>`: carrega a imagem com `stb_image`, faz o letterbox em C++, roda e lista as detecções.
   - libtorch vem do torch do `.venv` (`torch.utils.cmake_prefix_path`). Não há OpenCV C++ no sistema, por isso não há webcam em C++ nesta rodada.

## Validação (CA2)

- Entradas: `bus.jpg` e `zidane.jpg` (assets do Ultralytics) e 6 frames da webcam, capturados uma vez para `artifacts/data/` (fora do git).
- AOTI FP32 contra eager FP32: caixas com diferença máxima ≤ 0,5 px e scores ≤ 1e-3.
- AOTI FP16 contra eager FP16: caixas ≤ 2 px e scores ≤ 5e-3.
- Detecções com score ≥ 0,25: mesma quantidade, mesma classe e IoU ≥ 0,95 no pareamento.
- `detect` (pré-processamento em C++): em `bus.jpg`, as detecções com score ≥ 0,25 batem com as do eager em classe e IoU ≥ 0,90 (tolerância maior por diferenças de interpolação no resize).

## Benchmark (CA3)

- Variantes: eager Python FP32/FP16 e AOTI C++ FP32/FP16.
- Protocolo: só o forward, com o tensor já na GPU; 50 iterações de aquecimento e 500 medidas, cada uma sincronizada com o CUDA; nada mais usando a GPU; notebook na tomada.
- Métricas: latência média/p50/p95, throughput, memória do processo na GPU (NVML), pico de RSS, tamanho do `.pt2`, tempo de compilação, tempo de carga e da primeira execução.

## Relatório

`docs/aoti-results.md`, em pt-BR:

- versões (torch, CUDA usado no build e no runtime, driver, gcc, cmake) e os comandos para reproduzir;
- as dependências do binário (`ldd`) e onde ficam os pesos (embutidos no `.pt2` ou externos);
- operadores ou incompatibilidades encontrados;
- a tabela de resultados e uma conclusão, mesmo que o ganho seja nenhum.

## Riscos conhecidos

- O CUDA do sistema é 13.4, mas o torch é cu128. O build do AOTI e do C++ deve usar o CUDA 12.8 das wheels `nvidia-*` do `.venv`; se precisar do toolkit do sistema, documentar.
- GCC 16.2 é o único compilador C++ instalado; pode haver incompatibilidade com os headers do torch 2.11.
- O YOLOv10 pode não exportar limpo com `torch.export` (topk no pós-processamento, flags de export do Ultralytics).
