# PBI-097 — Engine próprio de YOLOv10s em CPU FP32

Design: [`2026-11-12-cpu-engine-design.md`](superpowers/specs/2026-11-12-cpu-engine-design.md).
Mesmo checkpoint fundido end-to-end das rodadas anteriores, batch 1, entrada
FP32 NCHW `1×3×640×640`. Corte somente antes da seleção top-300: o grafo
produz `[1,8400,84]`, caixas xyxy decodificadas e 80 scores sigmoid da cabeça
one-to-one. O pós-processamento C++ existente foi **movido, sem alteração**,
de `ncnn/cpp/postprocess.h` para `native/postprocess.h`; não há NMS.

**Resultado:** M1–M5 implementados, além das quatro correções do gerador
apontadas pela revisão independente. Referência em uma thread e otimizado em
uma/seis threads passam **8/8 FP32 estrito**, sem alterar referências ou
tolerâncias; `detect --check` passa. Nenhuma biblioteca de inferência, BLAS,
OpenMP/TBB ou dependência nova. No benchmark final (3 rodadas rotativas,
60 s de resfriamento, mesma sessão do OpenVINO), o engine fez **46,67 ms com
seis threads (21,4 FPS, ~465 GFLOPS, 58% do pico assumido)** contra
**43,71 ms do OpenVINO FP32** — 1,07× mais lento — e **215,81 ms com uma
thread**. Fica à frente das medianas históricas do eager (55,56 ms) e do ncnn
CPU (56,89 ms) e perto do ONNX Runtime (43,63 ms). A sessão foi feita com
navegador e chat abertos (não podiam ser fechados); por isso cada latência vem
acompanhada da margem até o "ideal" estimado pelo p5 (seção final).

## Construção e decisões

- `engine/export_raw_onnx.py`: usa `load_model('fp32').cpu().float()`, substitui
  `head.postprocess` pelo mesmo identity do exportador ncnn e exporta ONNX
  estático, opset 18, com constant folding. O helper existente passa
  transitoriamente por CUDA na carga; **a exportação/checagem do forward bruto
  é em CPU**. `--help` termina antes dos imports pesados e não exporta.
- `engine/gen/generate.py`: compilador offline, não interpretador. Infere
  formas, avalia constantes/Shape/Gather, confere todas as formas inferidas
  contra `onnx.shape_inference.infer_shapes` após materializar constantes e
  rejeita operadores, atributos e combinações não suportados. Não presume
  semântica de Resize: o grafo efetivamente contém
  `nearest/asymmetric/floor`, escalas `[1,1,2,2]`. MaxPool é 5×5, stride 1,
  padding 2 com −∞; Conv respeita grupos e strides, aceita padding simétrico
  e rejeita padding assimétrico explicitamente. Há também depthwise
  **7×7**, além das 3×3 citadas na estimativa inicial.
- Gera `model.cpp/model.h`, uma chamada por operação fundida, `plan.txt`
  com formas, MACs, aliases, intervalos de vida e offsets da arena, mais
  `weights.bin/weights.S`. Símbolo e offsets dos pesos alinhados em 64 bytes;
  `.incbin` em seção somente leitura. Não reorganiza pesos na inferência.
- Layout de ativações **NCHW contíguo**; atenção e head mantêm a ordem lógica
  ONNX. Isso permite slices contíguos e acesso vetorial à dimensão espacial,
  sem conversão global para NHWC/NCHWc. As transposições necessárias do PSA e
  do head são executadas e entram no tempo. Não se mediu layout blocado.
- **M5:** Conv 1×1 e 3×3 empacotam ativações diretamente em micro-painéis
  `[micro,K,16]`; não há im2col da imagem inteira. Kernel **6×16**, doze
  acumuladores AVX2/FMA, pesos `[ceil(OC/6),K,6]` com linhas finais zeradas,
  caudas de coluna com maskload/maskstore. C permanece nos registradores
  durante **todo K**; não se lê/escreve C a cada bloco de 128. Tiles completos
  usam loads/stores sem máscara. No pack de 1×1, deixa-se de saltar `H×W`
  floats entre cada acesso do GEMM. O caminho de cópia direta exige também
  `ih==oh && iw==ow`.
- Im2col usa segmentos de linha pré-calculados, cópias AVX2 contíguas em
  stride 1 e dois loads + shuffle/permute em stride 2. Divisão/módulo e
  limites são calculados antes do laço de canais. Só bordas ficam escalares;
  painéis completos não são zerados para depois serem sobrescritos.
- Grid **tile espacial × chunk de canais**, escolhido offline por forma:
  configuração final `NB=16`, chunks múltiplos de seis e equilibrados após
  aplicar o limite de cache. Busca pelo menos 24 jobs e painel B + fatia de
  pesos ≤1 MiB, deixando margem na L2 de 1,25 MiB. O maior B tem 288 KiB;
  não depende de toda a matriz de pesos permanecer na L1 de 48 KiB. Cada job
  empacota B uma vez; chunks distintos ainda repetem esse pack. A maior
  matriz densa deste grafo tem cerca de 1,5 MiB, não 9 MiB.
- Depthwise 3×3/7×7 usa halo zerado privado para vetorizar também as bordas
  de mapas pequenos. Stride 2 usa deinterleave, sem gather. Se há poucos
  canais, divide também por blocos de 16 linhas. MaxPool/Resize continuam
  AVX2; MatMul PSA usa bloco 4×16 sem alocar views por linha. Softmax vetoriza
  exp e, no DFL, oito reduções espaciais independentes.
- Broadcast binário coalesce a dimensão final e decodifica índices uma vez
  por sequência, não por float. Transposições PSA/head usam tiles 32×32 com
  micro-transposição AVX2 8×8 e caudas exatas; DFL troca blocos contíguos.
- Bias, SiLU e Add residual elegível ficam no epílogo. Apenas reduções
  semanticamente seguras são fundidas: uso único, mesmas formas e sem
  eliminar outros consumidores. Quinze Concat viram views, inclusive com
  produtores escrevendo diretamente em suas fatias. Reshape e slices/Split
  contíguos são aliases; Split não contíguo do PSA e concatenações não
  contíguas do head continuam copiando.
- Arena com planejamento de liveness, incluindo os consumidores de aliases
  e do residual fundido: **370.905.600 → 24.780.800 bytes**, redução de 93,3%.
  Chamadas geradas: **205 → 108**. Scratch de im2col é privado por worker,
  alocado na carga; descritores e algumas views temporárias ainda usam
  `std::vector`, portanto não se afirma inferência completamente sem heap.
- Pool próprio: N−1 threads criadas uma vez, mais a thread principal como
  worker. Afinidade via `pthread_setaffinity_np`, erros fatais; padrão de
  seis threads `0,2,4,6,8,10`, uma thread `0`. Tiles de saída disjuntos,
  publicação/completion por atomics release/acquire, espera ativa com
  `_mm_pause` entre operações; condvar entre inferências. Não há OpenMP/TBB.
- C++17, GCC **16.2.1**, `-O3 -march=x86-64-v3 -fno-math-errno`, sem
  `-ffast-math` global. Headers compartilhados de terceiros são includes de
  sistema; a compilação limpa não apresentou warnings no código do engine.
  `ldd`, salvo em `artifacts/engine/ldd.txt`, lista apenas vdso, libstdc++,
  libm, libgcc_s, libc e ld-linux; pthread está integrado à libc neste host.

### Exp e acúmulo

Todas as reduções são FP32. A ordem de soma pode diferir de PyTorch.
`exp8` usa redução por ln(2) com duas constantes e polinômio de grau 7 em
`[−ln(2)/2, ln(2)/2]`; erro de truncamento em aritmética exata < 7,1×10⁻⁹.
O teste de **1.000.001 amostras em [−88,88]**, comparado com `std::exp(double)`,
mediu erro relativo máximo **1,71849×10⁻⁷**, abaixo de 2×10⁻⁷. Escala dividida
preserva subnormais. Sigmoid usa exp de argumento não positivo, evitando
exp(+100); testes cobrem SiLU de −100 e −1000, incluindo zero negativo e
ausência de NaN. Em M5, SiLU usa `x/(1+exp(-x))` somente quando todas as
lanes estão em [−80,80], com o mesmo polinômio e escalas normais; fora dessa
faixa volta ao caminho estável com subnormais. O teste denso de 200.001
pontos em [−100,100] mediu erro relativo máximo **1,43031×10⁻⁷** onde a
magnitude é >10⁻³⁰ e também confere erro absoluto/relativo contra a referência
em toda a faixa. Não houve troca do exp de Softmax nem `-ffast-math` global.
O caminho referência continua usando `std::exp`.

## Marcos: medidas rápidas, sem proteção térmica

Entrada bus já preparada. Tempo = forward **mais** o pós-processamento
compartilhado, sem leitura, letterbox ou desenho. Primeira inferência
separada; percentis e impressão reutilizam `native/common.h`.

O gerador contou **10.851.302.400 MACs / 21.702.604.800 FLOPs**, incluindo as
convoluções e os dois MatMul — coincide com os 21,7026048 GFLOPs verificados
na rodada CPU anterior. Fórmula: `GFLOPS = 21,7026048 × 1000 / média_ms`.
Pico assumido fixo: **800 GFLOPS = aproximadamente 6×32 FLOP/ciclo×4,2 GHz**;
não é frequência medida nem utilização física exata. Uma thread também é
comparada ao pico total de seis núcleos, para manter a mesma base da tabela.

| Marco | Impl./threads | Aquecimentos + medidas | Média ms | p50 ms | p95 ms | GFLOPS | % de 800 | Validação |
|---|---|---:|---:|---:|---:|---:|---:|---|
| M1 — loops diretos | ref/1 | 0 + 2 | 21.692,22 | 21.692,22 | 21.826,33 | 1,00 | 0,13% | 8/8 |
| M2 — AVX2 + im2col/GEMM | opt/1 | 2 + 10 | 578,35 | 577,96 | 584,16 | 37,52 | 4,69% | 8/8 |
| M3 — fusões, views, arena | opt/1 | 2 + 10 | 347,54 | 343,59 | 364,22 | 62,45 | 7,81% | 8/8 |
| M4 — pool, final | opt/6 | 20 + 200 | **79,29** | 76,31 | 101,63 | **273,71** | **34,21%** | **8/8** |
| M4 — controle de uma thread | opt/1 | 20 + 200 | 348,13 | 347,31 | 352,82 | 62,34 | 7,79% | 8/8 |
| **M5 — verificação final, host ruidoso** | **opt/6** | **20 + 200** | **60,07** | **56,74** | **87,80** | **361,28** | **45,16%** | **8/8** |
| M5 — uma thread, comando solicitado | opt/1 | 20 + 50 | 240,15 | 238,91 | 248,97 | 90,37 | 11,30% | 8/8 |
| M5 — uma thread, comando original | opt/1 | 20 + 200 | 245,78 | 241,28 | 271,93 | 88,30 | 11,04% | 8/8 |

M2, M3, M4/6 e M5/6 incluem `--profile`; as medições de uma thread de
M4/M5 não. M1–M4 não tiveram cooldown nem monitor de concorrência. Na
verificação M5/6 houve uma espera de 60 s, mas sem monitor durante a medida;
isso **não tornou a execução livre de concorrência**. Houve atividade externa
no host, confirmada posteriormente pelos descartes do smoke. Não misturar
esses números com medianas térmicas nem atribuir causalidade isolada a cada
mudança de uma etapa que inclui várias otimizações. M4 mantém essencialmente
a latência de uma thread de M3 e dá **4,39×** de aceleração com seis threads
na sequência final, não 6× ideal.

Logs fora do Git, preservados em `artifacts/engine/`:
`m1-bench.txt`, `m2-bench.txt`, `m3-bench.txt`, `final-bench-{1t,6t}.txt`,
`m5-*.txt`; em particular `m5-final-bench-{6t,1t,1t-200}.txt`.
Os commits dos marcos preservam seus kernels históricos; `--stage 1..5`
seleciona as reescritas do gerador, **não restaura versões históricas dos
kernels**. A configuração padrão é M5; os pesos atuais são sempre 6×16.

### Perfil M4, seis threads

Médias das mesmas 200 inferências, timestamps por chamada gerada. As
porcentagens abaixo são do **tempo de modelo**, 78,66 ms; pós-processamento
é separado, **0,63 ms**, e a latência total é 79,29 ms. A diferença entre a
soma arredondada dos kernels e o modelo inclui instrumentação/dispatch.
`head` abrange as operações não Conv de DFL/decodificação/empacotamento.

| Tipo | Média ms | % do modelo |
|---|---:|---:|
| conv3x3 | 35,20 | 44,75% |
| conv3x3s2 | 8,82 | 11,21% |
| conv1x1 | 21,34 | 27,13% |
| dwconv | 4,71 | 5,99% |
| maxpool | 0,15 | 0,19% |
| resize | 0,20 | 0,25% |
| attention | 1,26 | 1,61% |
| concat/copy | 2,51 | 3,19% |
| head | 4,42 | 5,61% |
| other | 0,01 | 0,01% |

Carga M4/6: **2,62 ms**; primeira execução: **71,20 ms**; pico RSS:
**105,8 MiB**. M4/1: carga 1,03 ms, primeira 414,66 ms, RSS 101,0 MiB.
Carga mede arena, descritores e pool; pesos já estão no executável, sem
custo de leitura/repacking atribuído artificialmente ao forward. O blob de
pesos de M4 ocupava **58.087.488 bytes / 55,40 MiB**. Em M5 são
**58.570.624 bytes / 55,86 MiB**, por padding das linhas em múltiplos de seis.
Inclui original para `--impl ref` e cópia pré-empacotada para o otimizado.
Não é compressão do checkpoint.

### O que ajudou, o que não ajudou e o que não foi medido

- A troca dos loops diretos de M1 por im2col/GEMM foi a maior alavanca:
  21.692,22 → 578,35 ms, **37,5×**. Conv 3×3 ficou com im2col tiled, caminho
  validado e mensurável, sem o enorme intermediário da imagem inteira.
  **Não foi implementada uma segunda convolução direta blocada AVX2**;
  comparar com o loop direto escalar não prova superioridade universal de
  im2col sobre um kernel direto igualmente otimizado.
- Fusões, exp vetorial, views e melhor geração dos índices de im2col
  reduziram 578,35 → 347,54 ms (**39,9%**). O tempo de `other`, dominado
  anteriormente por ativações separadas, caiu de 143,00 para 0,09 ms;
  parte do trabalho migrou para o epílogo das Conv. Não tratar isso como
  143 ms eliminados sem custo. O scratch passou a calcular coordenadas
  espaciais uma vez por tile, não repetir divisão/módulo por peso/canal.
- Experimento adicional de painel espacial, **2 + 10, sem cooldown**, com
  o restante de M4 igual; logs `experiment-nb{32,64,128}-{1t,6t}.txt`:

  | NB | Média 1t ms | Média 6t ms | Decisão |
  |---|---:|---:|---|
  | 32 | 353,71 | 72,58 | Mais tiles, melhor 6t nesta sequência; pequeno custo 1t |
  | **64** | **350,19** | **77,14** | Compromisso mantido; B quente cabe na L1 |
  | 128 | 384,34 | 84,08 | Descartado: pior nos dois casos e scratch maior |

  NB=128 precisou dobrar a capacidade de scratch; o teste inicial rejeitou
  explicitamente capacidade insuficiente, sem escrever fora do buffer.
  NB=32 pode ser melhor para seis threads, mas a diferença entre 32 e 64
  não foi estabelecida sob controle térmico; **não há alegação de autotuning
  ótimo**. Os números históricos de M4 usam NB=64, restaurado ao final.
- NCHWc/NHWC e kernel direto AVX2 alternativo não foram comparados. Em M5
  foram comparados 4×16 e 6×16, conforme a seção seguinte. Não se atribui
  ganho a opções não implementadas. Winograd/FFT, FP16 e INT8 ficaram fora
  do escopo.

## Revisão do gerador — quatro correções antes de M5

A revisão independente identificou erros que não eram exercitados por este
grafo específico. Foram corrigidos em um commit separado e verificados com
grafos ONNX sintéticos, positivos e negativos:

1. **Ordem do residual:** em `A=Conv(in); B=Conv(in); Add(A,B)`, A não pode
   ler B antes de sua produção. A fusão exige produtor anterior à Conv;
   constantes/entrada já estão disponíveis. O teste
   `test_residual_future_producer_not_fused` impede a fusão em A e permite
   em B; `test_residual_earlier_producer_and_constant_fused` cobre constante.
2. **Padding assimétrico:** rejeitado com erro claro, pois o runtime recebe
   somente pads iniciais. `test_asymmetric_padding_rejected` e
   `test_symmetric_padding_accepted`. O pack rápido de 1×1 exige também
   igualdade das duas dimensões espaciais entre entrada e saída.
3. **Aliases da entrada:** Reshape/Slice/Split contíguos enraizados em
   `images` são rejeitados, em vez de deixar descritores nullptr sem rebind.
   `test_input_rooted_aliases_rejected` cobre as três operações;
   `test_arena_rooted_alias_accepted` mantém as views válidas da arena.
4. **Consumidores observáveis:** uso contado pela raiz dos aliases e também
   pela saída do grafo, não só pelas arestas presentes em `ops`. Conv→SiLU
   só funde uma ativação direta que seja o consumidor exclusivo. Testes
   `test_silu_alias_consumers_preserved` (Reshape/Slice/Split),
   `test_silu_graph_output_preserved` e `test_silu_sole_consumer_fused`.

Após essa parte, os três modos passaram 8/8, detect passou, CTest 3/3 e duas
regenerações tiveram hashes idênticos. M5 não alterou os loops de referência.

## M5 — medições por mudança

Todas as tentativas abaixo usaram `taskset -c 0,2,4,6,8,10`, **20 aquecimentos
+ 100 medidas e `--profile`**, exceto a linha final explicitamente 200.
Não são medianas térmicas nem um A/B isolado: a atividade externa cresceu
na segunda metade. `perf stat` foi tentado, mas **`perf: command not found`**
(exit 127); não se instalou pacote nem se usou sudo. Diagnóstico por
`--profile` e inspeção de assembly (`packed-assembly.txt`): os doze
acumuladores ficam nos registradores no laço K; os spills observados ficam
no epílogo, não a cada FMA.

| Mudança cumulativa / tentativa | Média 6t ms | p95 ms | Resultado / ressalva |
|---|---:|---:|---|
| Baseline após Parte A | 77,47 | 88,81 | Mesmo desempenho de M4 |
| Painéis `[micro,K,16]`, pack vetorial, K inteiro, kernel 4×16 | 71,21 | 74,74 | Conv3×3 caiu 34,59→25,85 ms; só empacotar 1×1 não bastou: 20,69→23,58 ms |
| Kernel 6×16 e linhas zeradas | 64,45 | 66,46 | Mantido; 1t 312,22→290,14 ms em 100 medidas |
| Grid 2-D por camada, tiles 64/32 | 60,80 | 67,73 | Mantido; conv1×1 20,68→18,82 ms |
| Broadcast por sequência + transposição blocada | 53,68 | 54,77 | Mantido; head 4,34→0,57 ms; concat/copy 2,52→0,14 ms |
| Halo depthwise + deinterleave | 53,98 | 60,21 | Kernel 4,34→2,25 ms; total contaminado por Conv mais lentas |
| Repetição do halo | 55,64 | 66,25 | Depthwise 2,39 ms; não esconder a repetição pior |
| Bounds/segmentos antes dos canais; sem zerar painel completo | 49,10 | 50,18 | Mantido; elimina divisões repetidas por canal |
| Loads/stores completos sem máscara | 48,95 | 50,37 | Ganho pequeno, não isolável do ruído |
| SiLU em faixa normal com fallback | 47,62 | 49,04 | Mantido, testado numericamente |
| NB=32 | 45,86 | 47,09 | Melhor que 64 nesta sequência |
| **NB=16 + chunks equilibrados** | **45,60** | **46,60** | Configuração mantida; ainda não <45 ms |
| NB=128, chunks equilibrados, NB=16 para K grande | 48,72 | 50,05 | **Descartado**, voltou para 16 |
| MatMul PSA 4×16 sem views por linha | 54,29 | 66,67 | Attention 1,26→0,62 ms; demais kernels ficaram mais lentos com o host ruidoso |
| Polinômio SiLU por Estrin | 51,64 | 60,29 | **Descartado**, ganho não demonstrado separadamente do ruído; voltou a Horner |
| Unroll K×4 explícito | 54,95 | 60,10 | **Descartado**, nenhum ganho demonstrado; voltou ao laço simples |
| **M5 final, 200 medidas, espera de 60 s, sem monitor** | **60,07** | **87,80** | Mantido como resultado final real, não substituído pelo melhor exploratório |

Os logs `m5-baseline.txt`, `m5-packed{4,6}.txt`, `m5-grid.txt`,
`m5-head.txt`, `m5-depthwise*.txt`, `m5-hoisted-pack.txt`, `m5-full-tile.txt`,
`m5-silu-normal.txt`, `m5-tile*.txt`, `m5-attention.txt`, `m5-estrin.txt`,
`m5-unroll4.txt` e `m5-final-*.txt` ficam em `artifacts/engine/`.

A comparação M5 1t final de **240,15 ms** contra M4 **348,13 ms** é redução
observada de 31,0%, mas também não foi uma sessão térmica comum. O resultado
6t final não permite concluir que a meta foi atingida: **60,07 ms** ficou
acima de 45 ms e dos runtimes históricos. Nem a melhor etapa de 100 medidas
(45,60 ms) passou o alvo. O smoke posterior confirmou carga externa suficiente
para esgotar seis tentativas de OV. Não é correto atribuir toda a diferença
a uma alteração de kernel nem garantir o resultado em host ocioso.

### Perfil M5 final, seis threads

Mesmas **200** chamadas da verificação final: **modelo 59,34 ms**, pós 0,73 ms,
total 60,07 ms. Percentuais do modelo; não da soma modelo+pós. Carga 2,85 ms,
primeira execução 57,65 ms, RSS máximo 107,6 MiB.

| Tipo | Média ms | % do modelo |
|---|---:|---:|
| conv3x3 | 24,89 | 41,94% |
| conv3x3s2 | 7,16 | 12,07% |
| conv1x1 | 22,35 | 37,66% |
| dwconv | 2,70 | 4,55% |
| maxpool | 0,20 | 0,34% |
| resize | 0,26 | 0,44% |
| attention | 0,72 | 1,21% |
| concat/copy | 0,19 | 0,32% |
| head | 0,80 | 1,35% |
| other | 0,02 | 0,03% |

Os slow paths removidos foram as transposições PSA (`Transpose`,
`Transpose_1`), DFL (`dfl/Transpose`) e saída (`model.23/Transpose`), além de
`model.23/Mul_2` com strides broadcast `[1,8400]`. O plano gerado mantém os
nomes, formas e grids. A matemática de âncoras/strides não foi aproximada
nem reassociada para reduzir operações. Convoluções densas agora são cerca
de 92% do modelo; atenção e head deixaram de ser gargalos dominantes.

## Correção e testes

Exportação: `onnx.checker(full_check=True)` PASS; uma chamada ORT CPU contra
`artifacts/ncnn/bus.raw.bin`: **max abs diff = 0,000915527344**. Nenhum
exportador antigo foi executado e nenhuma referência foi regenerada.

| Execução final | PASS | Raw bus max abs diff vs fixture ncnn | Bus: caixa / score máximo |
|---|---:|---:|---|
| `--impl ref --threads 1` | 8/8 | 0,000579834 | 0,000031 px / 0,000001 |
| `--impl opt --threads 1` | 8/8 | 0,000732422 | 0,000031 px / 0,000001 |
| `--impl opt --threads 6` | 8/8 | 0,000732422 | 0,000031 px / 0,000001 |

As tolerâncias continuam **caixas ≤ 0,5 px, scores ≤ 0,001, mesma contagem e
classe, IoU ≥ 0,95**, via `validate(infer, false, REF_DIR)` de `native/common.h`.
Alguns `overall_max` informativos são grandes, por permutações de linhas
TopK de baixa confiança; o matcher estrito não foi alterado. As oito entradas
não equivalem a uma avaliação COCO completa.

- `detect bus --check`: **PASS, 5/5, IoU mínimo 0,999281**, com letterbox C++.
- Runner ncnn recompilado após a mudança do include: **PASS postprocess bus
  bit-match (all 300 rows)**.
- Build Release limpo + CTest: **3/3** (`kernels`, `kernels_threads`,
  `thread_pool`). Conv 1×1/3×3, grupos e depthwise 3×3/7×7, stride 1/2,
  padding, canais/larguras ímpares, caudas M/N/K, bias/SiLU/residual,
  MaxPool com entradas negativas, Resize, MatMul, Softmax, transposição,
  broadcasting, exp e extremos de ativação. M5 acrescenta todas as caudas
  até 19 colunas, OC não múltiplo de seis, K>4608, grid 2-D, halos e blocos
  de linha com poucos canais, deinterleave nas bordas, transposições 8×8/
  32×32 ímpares e broadcasts por linha/coluna nos dois sentidos. Testes
  repetidos em 1 e 6 threads.
- Build separado `-fsanitize=thread`: **3/3**, sem relatos de data race.
  Pool testado em 1/2/6 threads, afinidade, barreiras, exactly-once,
  sleep/wake, trabalho vazio e tarefa única.
- `engine/tests/test_generator.py`: **13/13**, os quatro testes originais
  de rejeição mais nove testes sintéticos positivos/negativos das correções
  da Parte A, com subcasos para Reshape/Slice/Split e pads por eixo.
- Duas gerações consecutivas: **SHA-256 idêntico dos cinco arquivos**.

## Comparação com as bibliotecas

| Implementação | Latência ms | Protocolo |
|---|---:|---|
| OpenVINO FP32 (mesma sessão) | **43,71** (ideal ≈ 40,78) | Benchmark final abaixo, host com carga |
| OpenVINO FP32 (rodada CPU) | 40,20 | Mediana térmica, host ocioso |
| ONNX Runtime FP32 | 43,63 | Mediana térmica da rodada CPU, host ocioso |
| **Engine próprio, M5, 6 threads** | **46,67** (ideal ≈ 45,40) | Benchmark final abaixo, host com carga |
| PyTorch eager FP32 | 55,56 | Mediana térmica da rodada CPU, host ocioso |
| ncnn CPU FP32 | 56,89 | Mediana térmica da rodada ncnn, host ocioso |
| Engine próprio, M4, 6 threads | 79,29 | Medida rápida, sem controle térmico |

A única comparação controlada é com o OpenVINO na mesma sessão: **1,07×**
(46,67 vs 43,71 ms), ou 1,11× comparando os ideais estimados (45,40 vs 40,78).
As demais linhas são de sessões anteriores com host ocioso; como a sessão
final tinha carga externa, a vantagem do engine sobre eager e ncnn é, se algo,
subestimada. Conclusão: um engine escrito do zero, especializado no grafo e
sem bibliotecas, chega a ~93% da velocidade do OpenVINO em FP32 com AVX2.
O restante está quase todo nas convoluções densas (~92% do tempo): packing
repetido entre chunks de canais, epílogos e layout NCHW; convoluções diretas
em layout blocado (NCHWc8, como o oneDNN) seriam o próximo passo.

## Benchmark final (controle térmico, host com carga)

`BENCH_CONTENDED=1 bash scripts/bench_engine.sh` em 2026-11-12: três rodadas
com início rotativo, 60 s de resfriamento antes de cada execução, 20 + 200
iterações por execução, engine em `taskset -c 0,2,4,6,8,10` (6 threads) ou
`-c 0` (1 thread), OpenVINO FP32 exatamente como na rodada CPU. O navegador
(Helium) e o chat (Discord) **não puderam ser fechados**; em vez de descartar
as execuções, o modo `BENCH_CONTENDED=1` registra a carga externa medida pelo
ContentionMonitor (processos de build continuam descartando a execução).

Como a carga externa infla a média, cada linha traz uma estimativa do ideal:
o **p5** de cada execução (média das 5% iterações mais rápidas, as menos
afetadas), e a margem `média/ideal − 1`. É uma estimativa, não uma medida em
host ocioso.

| Variante | Mediana média ms | Ideal (p5) ms | Margem | Mediana p95 ms | FPS | Min/max média ms | CPU externa média |
|---|---:|---:|---:|---:|---:|---:|---:|
| **engine-6t** | **46,67** | 45,40 | +2,8% | 49,08 | 21,4 | 46,36/47,66 | 49,5% |
| engine-1t | 215,81 | 212,06 | +1,8% | 222,43 | 4,6 | 213,16/216,62 | 47,6% |
| ov-cpu-fp32 | 43,71 | 40,78 | +7,2% | 49,35 | 22,9 | 42,71/44,36 | 26,2% |

"CPU externa" é a soma do uso dos outros processos (100% = um núcleo); o OV
roda o monitor dentro do próprio processo, o engine num processo à parte, então
a coluna serve como ordem de grandeza. O ideal estimado do OpenVINO (40,78 ms)
bate com a mediana da rodada CPU em host ocioso (40,20 ms), o que dá alguma
confiança no método. O engine oscila menos sob carga (margem de 2,8% contra
7,2%), coerente com o particionamento estático e as threads fixadas.
Frequência média dos P-cores durante o OV: 3,93–3,99 GHz. Saídas:
`artifacts/bench/engine-runs/` e `artifacts/bench/engine-summary.txt`.

## Reprodução

Com os artefatos de referência existentes e o ambiente já instalado, na
raiz do repositório; não rodar os exportadores anteriores:

```bash
uv run python engine/export_raw_onnx.py --help
uv run python engine/export_raw_onnx.py
uv run python engine/gen/generate.py
sha256sum artifacts/engine/generated/* > /tmp/a
uv run python engine/gen/generate.py
sha256sum artifacts/engine/generated/* | diff - /tmp/a
rm -rf build
cmake --preset release
cmake --build --preset release
ctest --preset release
build/bin/yolo_engine validate artifacts/ref --impl ref --threads 1
build/bin/yolo_engine validate artifacts/ref --impl opt --threads 1
taskset -c 0,2,4,6,8,10 build/bin/yolo_engine validate artifacts/ref --impl opt --threads 6
build/bin/yolo_engine detect artifacts/data/bus.jpg --check artifacts/ref/bus.eager_dets.txt
taskset -c 0,2,4,6,8,10 build/bin/yolo_engine bench --threads 6 --iters 200 --profile
taskset -c 0 build/bin/yolo_engine bench --threads 1
taskset -c 0 build/bin/yolo_engine bench --threads 1 --iters 50
ldd build/bin/yolo_engine
uv run python engine/tests/test_generator.py
cmake --preset tsan
cmake --build --preset tsan
ctest --preset tsan
cmake --preset release
cmake --build --preset release
build/bin/yolo_ncnn check-postprocess artifacts/ncnn
BENCH_SMOKE=1 bash scripts/bench_engine.sh
bash scripts/bench_engine.sh                    # host ocioso
BENCH_CONTENDED=1 bash scripts/bench_engine.sh  # host com carga: registra em vez de descartar
```

Para outro diretório gerado, usar `generate.py --output DIRETÓRIO` e
`cmake ... -DENGINE_GENERATED_DIR=DIRETÓRIO_ABSOLUTO`. Somente o conjunto de
operadores/atributos descrito pelo compilador é aceito; não é um runtime
ONNX geral. Entradas dinâmicas, outros modelos, batch > 1 e outras CPUs de
afinidade padrão não foram validados. Os testes TSan cobrem kernels/pool,
não um forward completo instrumentado de todas as oito imagens. A
aproximação de exp tem teste denso, não prova formal para todos os floats.

Próximos passos: repetir a sessão final em host ocioso; confirmar NB=16
contra 32/64 sem concorrência; reduzir o packing repetido entre chunks de
canais e melhorar os epílogos; depois testar convolução direta em layout
blocado (NCHWc8). A meta de 45 ms ficou na fronteira: 46,67 ms medidos com
carga externa, ~45,4 ms estimados sem ela. **INT8 com AVX-VNNI = PBI-098**,
com calibração e validação próprias; não foi implementado neste cartão.
