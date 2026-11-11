# PBI-095, rodada 3: baseline da YOLOv10s em CPU

Data: 2026-11-11 · Card: PBI-095 (Todoist `6hxmXq88f3mwWRf6`) · Status: design aprovado · Rodadas anteriores: AOTInductor e TensorRT na GPU

## Objetivo

Descobrir o desempenho de referência da YOLOv10s em CPU com runtimes prontos (ONNX Runtime e OpenVINO, FP32 e INT8), comparado com o PyTorch eager e com o teto teórico da CPU. É o passo 1 antes de um engine próprio em C++, que será um projeto separado.

## Decisões (com o Nicolas, 2026-11-11)

- **FP16 fica fora**: o i5-13450HX tem AVX2, FMA, F16C e AVX-VNNI, mas não AVX-512 FP16 nem AMX; FP16 não acelera a conta nessa CPU. O tipo reduzido que interessa é **INT8 com AVX-VNNI**.
- Benchmark em **Python**: com inferências de dezenas de ms, o custo da chamada Python é desprezível, e o objetivo desta rodada é achar o alvo de desempenho.
- **Threads**: variantes principais com 6 threads, uma por P-core (`taskset -c 0,2,4,6,8,10`; P-cores são as CPUs 0–11 com HT, E-cores 12–15), e o mesmo número configurado no runtime. Uma variante extra com o padrão do OpenVINO, sem fixar nada, mostra o efeito dos núcleos híbridos.
- Mesma entrada estática 1×3×640×640 e mesmo ONNX FP32 da rodada 2 (`artifacts/yolov10s_fp32.onnx`).

## Variantes

1. `eager-cpu fp32`: PyTorch eager na CPU (`torch.set_num_threads(6)`).
2. `ort-cpu fp32`: ONNX Runtime, CPU Execution Provider, otimizações de grafo completas.
3. `ov-cpu fp32`: OpenVINO, dica de precisão f32, dica de desempenho LATENCY, 6 threads.
4. `ov-cpu int8`: OpenVINO com quantização pós-treino do NNCF.
5. `ov-cpu fp32 default`: OpenVINO sem configurar threads nem afinidade.

## Validação

- FP32 (variantes 1, 2, 3 e 5): mesmas referências (`artifacts/ref/`) e tolerâncias estritas das rodadas anteriores (caixas até 0,5 px, score até 0,001, detecções com score ≥ 0,25 com mesma quantidade, mesma classe e IoU ≥ 0,95), com diferenças medidas sobre pares casados por IoU.
- INT8, critério de detecção nas 8 imagens, contra o eager FP32: mesma classe, IoU ≥ 0,80 e diferença de score ≤ 0,05 nos pares. Uma detecção sem par só é aceita se o seu score for menor que 0,30 (limiar 0,25 + tolerância de score), porque uma detecção na borda pode cruzar o limiar legitimamente.
- INT8, critério de acurácia: mAP50-95 (pycocotools) nas **500 primeiras imagens do COCO val2017** (ordenadas por id), INT8 contra OpenVINO FP32 nas mesmas imagens. Perda aceita: até **1,0 ponto**. O eager FP32 também é avaliado, como sanidade.
- Calibração do INT8: **300 imagens do val2017 de 501 a 800** na mesma ordem, disjuntas das de avaliação. (Ajuste em relação à conversa: baixar o train2017 custaria ~18 GB.)

## Benchmark

- Só a inferência, com o tensor de entrada já pronto em memória; 20 iterações de aquecimento e 200 medidas; cada variante no próprio processo.
- Métricas: latência média, p50 e p95, FPS, pico de RSS, tempo de carga e da primeira inferência, tamanho do modelo; mAP para FP32 e INT8 do OpenVINO e eager.
- Comparação com o teto: GFLOPs do modelo (via Ultralytics) dividido pela latência, contra o pico teórico FP32 dos 6 P-cores (32 FLOP/ciclo por núcleo com 2 FMAs AVX2 de 256 bits, na frequência all-core medida durante a execução).

## Entregas

- Scripts em `cpu/`, `native/bench_cpu.sh` para reproduzir e `docs/cpu-results.md` em pt-BR com versões, comandos, tabela, validação, mAP, comparação com o teto e conclusão.

## Riscos conhecidos

- NNCF pode quantizar o pós-processamento do head (TopK, gather) e piorar a saída; pode ser preciso um `ignored_scope`.
- A CPU de notebook muda de frequência com a temperatura; as variantes precisam rodar em sequência na mesma sessão.
