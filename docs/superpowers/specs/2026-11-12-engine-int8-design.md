# PBI-098: INT8 com AVX-VNNI no engine C++ próprio

Data: 2026-11-12 · Card: PBI-098 (Todoist `6hxx9w92QWrp4956`) · Status: design aprovado (autorização do Nicolas para seguir) · Antecede: PBI-097 (engine FP32)

## Objetivo

Levar o engine próprio para INT8 usando AVX-VNNI (`vpdpbusd`: 8 lanes × 4 produtos u8×s8 por instrução, contra 8 FMAs FP32), com quantização e calibração feitas por nós, e comparar com o OpenVINO INT8 (NNCF): 14,91 ms na rodada 3, mAP50-95 49,33 → 49,02.

Ponto de partida (PBI-097): engine FP32 46,67 ms com 6 threads; as convoluções densas são ~92% do tempo. O ganho teórico do VNNI sobre FMA é 4× por instrução, então o alvo realista é **abaixo de 20 ms**, e a meta esticada é **empatar com o OpenVINO INT8 na mesma sessão**.

## Decisões

1. **Esquema de quantização** (o mesmo do preset `mixed` do NNCF usado na rodada 3):
   - pesos: INT8 simétrico **por canal de saída** (s8 em [−127, 127]);
   - ativações: u8 assimétrico **por tensor**, com escala e zero-point vindos da calibração.

   O `vpdpbusd` multiplica u8 (ativação) por s8 (peso) e acumula em int32 sem saturação intermediária, então não há o problema de overflow do `pmaddubsw`.
2. **Quantizar no empacotamento.** As ativações continuam em FP32 na arena e todas as operações que não são convolução densa ficam em FP32 como hoje: SiLU, concat sem cópia, MaxPool, Resize, atenção, head e depthwise. A conversão FP32 → u8 acontece dentro do empacotamento do painel B, que já copia os dados hoje. O epílogo faz a desquantização (int32 × escala_a × escala_w[oc], correção do zero-point pré-calculada por canal, bias), aplica a SiLU e o residual e grava em FP32. Isso troca um pouco de banda de memória por simplicidade: nada muda no planejador de arena, nos views nem nos kernels FP32.
3. **Padding com o zero-point.** No im2col, o zero real vira `q = zero_point`, não 0. Uma alternativa equivalente é corrigir a soma no epílogo.
4. **Calibração própria**: 300 imagens COCO val, posições 501–800 por image id (as mesmas da rodada 3 e disjuntas das 500 de avaliação), com o mesmo pré-processamento (`cpu/common.read_image`). Os intervalos são o min/max de cada tensor que entra numa convolução densa. Se a acurácia não passar, tenta-se percentil (99,99%) antes de deixar camadas em FP32.
5. **Fallback por camada.** O gerador aceita uma lista de convoluções que ficam em FP32. Começa com todas as densas em INT8 (a depthwise fica sempre em FP32). Se o critério falhar, as candidatas a voltar pra FP32 são a primeira convolução (entrada da imagem) e as convoluções finais de caixa e classe do head. O relatório lista o que ficou em FP32 e por quê.
6. **Um binário, duas precisões.** O gerador emite as duas versões dos pesos e `yolo_engine` ganha `--precision fp32|int8`. O FP32 continua idêntico ao do PBI-097.

## Validação

- **FP32:** inalterado, 8/8 estrito nos três modos.
- **INT8 nas 8 imagens de referência:** o critério da rodada 3, `cpu/common.compare(..., int8=True)` contra a referência eager FP32: pares por IoU ≥ 0,80, diferença de score ≤ 0,05, detecções não pareadas só com score < 0,30. Feito em Python, chamando o engine por um modo `serve` (entrada e saída binárias por stdin/stdout).
- **mAP50-95** nas mesmas 500 imagens e com o mesmo código da rodada 3: queda ≤ 1,0 ponto contra o engine FP32 (o eager deu 49,33).
- **Kernels:** teste unitário do kernel VNNI contra a referência com inteiros exatos (bit a bit em int32), mais as caudas, o padding com zero-point e o epílogo.

## Benchmark

`native/bench_engine.sh` com engine FP32 6t, engine INT8 6t, OpenVINO FP32 e OpenVINO INT8 na mesma sessão. O protocolo térmico é o mesmo. O modo `BENCH_CONTENDED=1` vale quando o host não pode ficar ocioso, com ideal estimado pelo p5. `--profile` separa o tempo de empacotar e quantizar do tempo de GEMM.

## Entregáveis

Calibração e quantização em `engine/quant/`, kernel VNNI em `engine/runtime/`, os modos `serve` e `--precision` no `yolo_engine`, validação e mAP em `engine/`, e o relatório `docs/engine-int8-results.md` (pt-BR). Também README e REPRODUCE (inglês), uma seção nova no relatório PDF/HTML e um comentário no card.
