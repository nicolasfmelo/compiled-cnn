# PBI-099: TensorRT FP16 otimizado para a YOLOv10s na GPU

Data: 2026-11-16 · Card: PBI-099 (Todoist `6j32f5WgwwFf9XR6`) · Status: design aprovado em 16/11/2026 · Rodada de origem: `2026-11-11-tensorrt-gpu-design.md`

## Por quê

A rodada 2 do PBI-095 mediu o TensorRT FP16 em 2,94 ms com as opções padrão: builder no nível de otimização 3, `enqueueV3` simples e entrada quadrada 640×640. A pergunta agora é quanto dessa latência ainda dá para tirar **sem mudar precisão nem pesos**. INT8 e esparsidade 2:4 ficam para cards próprios.

## Objetivos

- Medir, em degraus e na mesma sessão, o efeito de cada alavanca sobre o TensorRT FP16:
  1. **A**, a baseline: 640×640, nível de otimização padrão (3), `enqueueV3`.
  2. **B**: 640×640 com `builderOptimizationLevel` 5.
  3. **C**: B mais a inferência por **CUDA Graph** (captura de `enqueueV3`).
  4. **D**: C com entrada **640×384**, o "perfil webcam".
- Entender onde está o tempo, com o perfil por camada do TensorRT para A e D.
- Mostrar que a entrada retangular não muda as detecções em frames 16:9.

## Não objetivos

- INT8, esparsidade 2:4, kernels CUDA próprios.
- FP32, AOTInductor e ncnn em 640×384.
- O pipeline da webcam (pré-processamento, cópias, desenho).
- Mudar as referências de 640×640, as tolerâncias ou o comportamento dos outros runners.

## Restrições

- TensorRT 11.4 e CUDA 13.4 do sistema, a mesma RTX 3050 Laptop, sem clocks travados (cerca de ±3% entre execuções).
- Os artefatos (`artifacts/`, `yolov10s.pt`) não estão versionados. Ficam no checkout principal e são copiados ou ligados como na rodada 2 (`docs/tensorrt-results.md:410`). Tudo o que esta rodada gera vai para nomes novos, sem sobrescrever os arquivos de 640×640 existentes.
- Os resultados de D mudam a carga de trabalho e não entram na tabela principal do README (640×640). Ficam numa seção à parte, como "perfil webcam".

## Abordagens consideradas

1. **Ablação em degraus com o perfil webcam separado** (escolhida). Cada alavanca tem o seu efeito medido. B e C continuam comparáveis com as outras linhas do README, e D é declarado como outra carga de trabalho.
2. **Só a combinação final contra a baseline.** Menos medições, mas não diz o que rendeu, e mistura a mudança de carga com a otimização.
3. **Formato dinâmico (perfil de otimização com faixa de formatos)** em vez de engines estáticos. É mais flexível, mas os kernels do TensorRT para formato dinâmico tendem a ser mais lentos que os estáticos, o que trabalha contra o objetivo. Fica fora.

## Decisões

- **Só FP16.** O ONNX vem do mesmo exportador em half, com o engine *strongly typed* (`--stronglyTyped`).
- **Entrada retangular:** altura 384 e largura 640 (NCHW `1×3×384×640`), letterbox com `gain = min(640/w, 384/h)` e a mesma regra de arredondamento e padding cinza 114 do letterbox atual. Num frame 1280×720, a imagem fica com 640×360 e 12 px de padding em cima e embaixo: **a mesma resolução efetiva** do 640×640.
- **Referências novas do PyTorch eager FP16 em 384×640**, geradas das mesmas 8 imagens (`artifacts/data/`) em `artifacts/ref_384x640/`, no mesmo formato de `artifacts/ref/`. As de 640×640 não são tocadas.
- **Validação:**
  - TensorRT contra eager em cada formato, com as tolerâncias FP16 atuais (caixas ≤ 2 px, scores ≤ 5e-3, pares casados por IoU ≥ 0,95 entre detecções com score ≥ 0,25, mesma contagem e mesma classe), para A, B, C e D;
  - **retangular contra quadrado** nas 7 imagens 16:9 (zidane e os 6 frames da webcam): as detecções de D e de C, nas coordenadas da imagem original, devem casar com mesma contagem e classe, IoU ≥ 0,90 e score com diferença ≤ 0,02. O bus é retrato e encolhe em 384 de altura, então fica de fora dessa comparação. Os critérios dessa comparação são decisão minha: o padding diferente muda o contexto das convoluções nas bordas, então não se espera igualdade exata. Se falhar, o relatório mostra os pares e a diferença, e a falha não é escondida.
- **CUDA Graph:** capturar `enqueueV3` uma vez, depois de uma execução normal de aquecimento, e medir `cudaGraphLaunch` mais a sincronização. Os endereços de entrada e saída são fixos, então o grafo não precisa ser recapturado.
- **Duas medidas por iteração:** o relógio do host (como hoje: do lançamento até a sincronização) e o tempo de GPU por CUDA events em volta do lançamento. O relatório mostra as duas, para separar o ganho de lançamento do ganho de kernel.
- **Variação entre builds:** o TensorRT pode escolher táticas diferentes a cada build (`docs/tensorrt-results.md:251`). A e B são construídos **duas vezes cada** e os dois builds de cada um são medidos. Assim, o efeito do nível 5 é comparado com a variação de um build para o outro.
- **Benchmark:** variantes A (2 builds), B (2 builds), C, D e o eager FP16 em 640×640 e em 384×640 (para dar o speedup em cada formato). Três rodadas com ordem rotativa, 30 s de pausa antes de cada execução e mediana das três. O protocolo por execução continua o da rodada 1: entrada já na GPU, 50 aquecimentos, 500 medidas.
- **Perfil por camada:** `trtexec --dumpProfile --separateProfileRun` nos engines de A e D, com as 10 camadas mais caras de cada um no relatório. Se o perfil exigir `--profilingVerbosity=detailed` no build, ele é feito num engine construído só para isso, e o relatório diz que esse engine não é o medido.

## O que fica para quem implementa

- Como generalizar `INPUT_SIZE` e o letterbox de `native/common.h` para o formato do engine sem mudar o comportamento dos outros runners (`aoti`, `ncnn`, `engine` continuam em 640×640).
- Nomes de arquivos e de variantes, a interface de linha de comando do runner (flag de grafo e formato lido do engine) e o script de benchmark.
- O ponto exato da captura do grafo e o tratamento de erro.

## Entrega

- Relatório técnico `docs/tensorrt-fp16-tuning-results.md`, em pt-BR: versões, comandos, tempo de build por nível, validação, tabela da ablação (host e GPU, p50/p95), variação entre builds, perfil por camada, comparação retangular contra quadrado e conclusão.
- README e `docs/REPRODUCE.md` em inglês: a seção "perfil webcam" e a linha de C (640×640) junto com o TensorRT, mais os passos de reprodução.
- Relatório final PDF/HTML (`docs/report/`, pt-BR): seção nova com a ablação.
