# PBI-097: engine de inferência próprio em C++ para a YOLOv10s na CPU

Data: 2026-11-12 · Card: PBI-097 (Todoist `6hxr6h6wmgQqwHwc`) · Status: design aprovado · Antecede: PBI-095 (rodadas 1–4)

## Objetivo

Responder se um engine escrito do zero, especializado neste grafo e sem nenhuma biblioteca de inferência, chega perto do OpenVINO na CPU. Referências da rodada 3 (medianas com controle térmico, 6 threads nos P-cores): eager 55,56 ms, ONNX Runtime 43,63 ms, OpenVINO FP32 40,20 ms (~70% do pico FP32), ncnn CPU 56,89 ms.

## Grafo (análise de `artifacts/yolov10s_fp32.onnx`)

- 10,79 GMACs por imagem (21,6 GFLOPs). Convoluções 3×3 comuns: 26 camadas, 6,18 GMACs (57%). Convoluções 1×1: 44 camadas, 4,56 GMACs (42%). Depthwise: 16 camadas, 0,05 GMACs (0,5%), limitadas por memória.
- Demais operações: SiLU (Sigmoid+Mul), Concat, Split, MaxPool (SPPF), Resize ×2, atenção do PSA (MatMul ×2, Softmax), decodificação do head. Pesos: 27,7 MiB.
- Pico teórico FP32 da CPU (6 P-cores, 2 FMA AVX2 de 8 floats por ciclo): ≈ 6 × 32 × 4,2 GHz ≈ 800 GFLOPS. 40 ms equivalem a ~540 GFLOPS efetivos.

## Decisões (aprovadas pelo Nicolas em 2026-11-12)

1. **Gerador de código, não interpretador.** Um script Python lê o ONNX e gera C++ especializado: formatos fixos, buffers planejados na geração, chamadas diretas aos kernels. Os pesos ficam num blob embutido no executável (`.incbin`). O binário depende só de libc, libstdc++, libm e pthreads: sem BLAS, sem runtime de inferência.
2. **Corte do grafo no head bruto** `[1, 8400, 84]`, como na rodada 4 (ncnn). A seleção top-300 reaproveita o pós-processamento em C++ que bate bit a bit com o Python, movido de `ncnn/cpp/` para `native/`.
3. **Marcos**, cada um commitado, validado e medido:
   - **M1, referência:** loops diretos FP32 em uma thread; só correção. Continua disponível como oráculo (`--impl ref`).
   - **M2, kernels:** convolução como GEMM com pesos reorganizados na carga; micro-kernel AVX2+FMA (bloco de registradores ~6×16); 1×1 como GEMM direto; 3×3 via im2col (ou convolução direta, se medido melhor); kernel próprio de depthwise. Teste unitário de cada kernel contra a referência.
   - **M3, fusões e memória:** bias + SiLU no epílogo do GEMM (exp vetorizado), concat/split sem cópia (escrita direto na fatia de destino), layout blocado se a medição mostrar ganho.
   - **M4, threads:** pool próprio fixado nos P-cores (CPUs 0,2,4,6,8,10), divisão por blocos da saída, sincronização por spin entre camadas.
4. **INT8 com VNNI fica para um PBI separado** (PBI-098).
5. **Critério de sucesso:** M1–M4 completos e validados, com o resultado reportado honestamente. Vencer os 40,20 ms do OpenVINO é meta, não condição de aceite.

## Validação

- Mesmas referências (`artifacts/ref/`), mesmas 8 entradas e mesmas tolerâncias FP32 estritas (caixas ≤ 0,5 px, score ≤ 0,001, mesma contagem e classe, IoU ≥ 0,95), via `validate()` de `native/common.h`. Nenhuma tolerância é afrouxada.
- `detect --check` em `bus.jpg` com letterbox em C++.
- Saída bruta comparada com `artifacts/ncnn/bus.raw.bin` (diferença máxima informada).
- Testes unitários dos kernels (otimizado vs referência) com formatos ímpares e bordas.

## Benchmark

`native/bench_engine.sh`: mesmo protocolo térmico das rodadas 3/4 (resfriamento de 60 s, 3 rodadas rotativas, mediana, monitor de contenção), com o engine em 6 threads, o engine em 1 thread e o OpenVINO FP32 na mesma sessão. Relatório com GFLOPS efetivos, percentual do pico e tempo por tipo de camada (`--profile`). Compilação `-O3 -march=x86-64-v3` (AVX2/FMA/F16C), portátil para qualquer CPU com AVX2.

## Entregáveis

`engine/` (exportação do ONNX bruto, gerador, runtime e kernels, CMake, testes), `native/bench_engine.sh`, `docs/engine-results.md` (pt-BR), README e REPRODUCE em inglês atualizados, relatório final PDF/HTML atualizado e comentário no card.
