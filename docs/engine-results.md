# PBI-097 — Engine próprio de YOLOv10s em CPU FP32

Design: [`2026-11-12-cpu-engine-design.md`](superpowers/specs/2026-11-12-cpu-engine-design.md).
Mesmo checkpoint fundido end-to-end das rodadas anteriores, batch 1, entrada
FP32 NCHW `1×3×640×640`. Corte somente antes da seleção top-300: o grafo
produz `[1,8400,84]`, caixas xyxy decodificadas e 80 scores sigmoid da cabeça
one-to-one. O pós-processamento C++ existente foi **movido, sem alteração**,
de `ncnn/cpp/postprocess.h` para `native/postprocess.h`; não há NMS.

**Resultado:** M1–M4 implementados e commitados separadamente. Referência em
uma thread e otimizado em uma/seis threads passam **8/8 FP32 estrito**, sem
alterar referências ou tolerâncias. `detect --check` passa. O engine não usa
runtime de inferência, BLAS, OpenMP, TBB ou nova dependência. A medição rápida
final foi **79,29 ms / 273,71 GFLOPS em seis P-cores**, ainda atrás das
bibliotecas de referência. **Não é um benchmark final com controle térmico.**

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
  padding 2 com −∞; Conv respeita grupos, strides e pads. Há também depthwise
  **7×7**, além das 3×3 citadas na estimativa inicial.
- Gera `model.cpp/model.h`, uma chamada por operação fundida, `plan.txt`
  com formas, MACs, aliases, intervalos de vida e offsets da arena, mais
  `weights.bin/weights.S`. Símbolo e offsets dos pesos alinhados em 64 bytes;
  `.incbin` em seção somente leitura. Não reorganiza pesos na inferência.
- Layout de ativações **NCHW contíguo**; atenção e head mantêm a ordem lógica
  ONNX. Isso permite slices contíguos e acesso vetorial à dimensão espacial,
  sem conversão global para NHWC/NCHWc. As transposições necessárias do PSA e
  do head são executadas e entram no tempo. Não se mediu layout blocado.
- Conv 1×1 usa GEMM diretamente sobre a entrada; Conv 3×3 usa **im2col em
  tiles + GEMM**, sem materializar o im2col da imagem inteira. Microkernel
  **4 canais × 16 posições**, oito acumuladores AVX2/FMA, pesos
  `[ceil(OC/4),K,4]`, caudas escalares. `NB=64`, `KB=128`: painel quente B de
  32 KiB + pesos/saída cabem na L1d de 48 KiB. O maior painel completo é
  `4608×64×4 = 1,125 MiB`, abaixo dos 1,25 MiB de L2 por P-core; é uma margem
  pequena para outras estruturas, não uma promessa de ausência de misses.
- Depthwise 3×3/7×7 tem kernel AVX2 próprio, bordas escalares e gather no
  stride 2. MaxPool, Resize e MatMul de atenção têm caminhos AVX2; Softmax
  vetoriza exp e, no DFL, oito reduções espaciais independentes.
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
ausência de NaN. O caminho referência continua usando `std::exp`.

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

M2, M3 e M4/6 incluem instrumentação `--profile`; M4/1 não. As medições
não tiveram cooldown nem monitor de concorrência. Houve atividade externa
no host, confirmada posteriormente pelos descartes do smoke. Não misturar
esses números com medianas térmicas nem atribuir causalidade isolada a cada
mudança de uma etapa que inclui várias otimizações. M4 mantém essencialmente
a latência de uma thread de M3 e dá **4,39×** de aceleração com seis threads
na sequência final, não 6× ideal.

Logs fora do Git, preservados em `artifacts/engine/`:
`m1-bench.txt`, `m2-bench.txt`, `m3-bench.txt`, `final-bench-{1t,6t}.txt`.
Os commits dos marcos preservam seus kernels históricos; `--stage 1..4`
seleciona as reescritas do gerador, **não restaura versões históricas dos
kernels**. A configuração padrão do gerador é M4.

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
pesos ocupa **58.087.488 bytes / 55,40 MiB**: inclui original para `--impl ref`
e cópia pré-empacotada para o otimizado. Não é compressão do checkpoint.

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
- NCHWc/NHWC, kernel direto AVX2 alternativo e outras formas de microkernel
  não foram comparados. Não se atribui ganho a opções não implementadas.
  Winograd/FFT, FP16 e INT8 ficaram fora do escopo.

## Correção e testes

Exportação: `onnx.checker(full_check=True)` PASS; uma chamada ORT CPU contra
`artifacts/ncnn/bus.raw.bin`: **max abs diff = 0,000915527344**. Nenhum
exportador antigo foi executado e nenhuma referência foi regenerada.

| Execução final | PASS | Raw bus max abs diff vs fixture ncnn | Bus: caixa / score máximo |
|---|---:|---:|---|
| `--impl ref --threads 1` | 8/8 | 0,000579834 | 0,000031 px / 0,000001 |
| `--impl opt --threads 1` | 8/8 | 0,000762939 | 0,000031 px / 0,000000 |
| `--impl opt --threads 6` | 8/8 | 0,000762939 | 0,000031 px / 0,000000 |

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
  broadcasting, exp e extremos de ativação. Testes repetidos em 1 e 6 threads.
- Build separado `-fsanitize=thread`: **3/3**, sem relatos de data race.
  Pool testado em 1/2/6 threads, afinidade, barreiras, exactly-once,
  sleep/wake, trabalho vazio e tarefa única.
- `engine/tests/test_generator.py`: **4/4**, rejeita operador válido ONNX
  não suportado, atributo não suportado, domínio customizado e modo de
  coordenadas Resize não suportado, sem produzir pesos parciais.
- Duas gerações consecutivas: **SHA-256 idêntico dos cinco arquivos**.

## Comparação com as bibliotecas

Referências históricas de seis P-cores fornecidas pelo projeto, **não
reexecutadas como benchmark térmico nesta tarefa**:

| Implementação | Média / mediana ms | Protocolo |
|---|---:|---|
| OpenVINO FP32 | 40,20 | Mediana térmica da rodada CPU |
| ONNX Runtime FP32 | 43,63 | Mediana térmica da rodada CPU |
| PyTorch eager FP32 | 55,56 | Mediana térmica da rodada CPU |
| ncnn CPU FP32 | 56,89 | Mediana térmica da rodada ncnn |
| **Engine próprio, M4/6** | **79,29** | **Uma execução rápida, sem controle térmico** |

A razão meramente descritiva das latências é 1,97× vs OV, 1,82× vs ORT,
1,43× vs eager e 1,39× vs ncnn; **protocolos diferentes, não speedups
controlados**. A conclusão segura é que remover bibliotecas/Python não
bastou para superar kernels maduros. Conv densas ainda consomem cerca de
83% do modelo; head/transposições, depthwise, cache e balanceamento de tiles
limitam a eficiência, e o teto ideal de 800 GFLOPS não é todo atingível.

## Benchmark final (controle térmico)

**PENDENTE — preencher pelo coordenador, em sessão ociosa. Não há números
finais nesta seção. Não foi executada a sessão completa de três rodadas.**

```bash
bash native/bench_engine.sh
```

Protocolo implementado: 60 s antes de **cada tentativa**, check-host de 5 s
inalterado, ContentionMonitor durante o nativo excluindo apenas o filho
medido, seis tentativas máximas e retry somente em exit 75. Três rodadas,
inícios rotativos 0/1/2, variantes `engine-6t`, `engine-1t`, `ov-cpu-fp32`.
OV usa exatamente o runner/propriedades/20+200/afinidade do benchmark CPU
existente, com seu próprio monitor. Cada engine usa 20+200. Resumo calcula
medianas de médias, p95 e FPS por execução, além de min/max das médias;
não mistura as 600 iterações. Saídas próprias:
`artifacts/bench/engine-runs/`, `artifacts/bench/engine-summary.txt`.

| Variante | Mediana média ms | Mediana p95 ms | Mediana FPS | Min/max média ms |
|---|---|---|---|---|
| engine-6t | PENDENTE | PENDENTE | PENDENTE | PENDENTE |
| engine-1t | PENDENTE | PENDENTE | PENDENTE | PENDENTE |
| ov-cpu-fp32 | PENDENTE | PENDENTE | PENDENTE | PENDENTE |

**Somente smoke executado:** `BENCH_SMOKE=1 bash native/bench_engine.sh`,
exit 0, uma rodada, cooldown de 5 s, engine 1+5, OV ainda 20+200. Arquivos em
`artifacts/bench/engine-runs/smoke/`, claramente rotulados, sem substituir o
resumo final. Foram necessários **oito descartes por concorrência**:
três no host check de engine-6t, um durante engine-1t e outro no seu host
check, dois durante OV e um no seu host check. O script retentou com novo
cooldown e preservou o log; não houve relaxamento dos limiares.
Resultados aceitos de smoke: 72,50 / 348,67 / 43,48 ms, respectivamente.
São teste de protocolo/parser, **não resultados térmicos**.

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
rm -rf engine/build
cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build
ctest --test-dir engine/build --output-on-failure
engine/build/yolo_engine validate artifacts/ref --impl ref --threads 1
engine/build/yolo_engine validate artifacts/ref --impl opt --threads 1
taskset -c 0,2,4,6,8,10 engine/build/yolo_engine validate artifacts/ref --impl opt --threads 6
engine/build/yolo_engine detect artifacts/data/bus.jpg --check artifacts/ref/bus.eager_dets.txt
taskset -c 0,2,4,6,8,10 engine/build/yolo_engine bench --threads 6 --profile
taskset -c 0 engine/build/yolo_engine bench --threads 1
ldd engine/build/yolo_engine
uv run python engine/tests/test_generator.py
cmake -S engine -B engine/build-tsan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENGINE_TSAN=ON
cmake --build engine/build-tsan --target test_kernels test_thread_pool
ctest --test-dir engine/build-tsan --output-on-failure
cmake -S ncnn/cpp -B ncnn/cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ncnn/cpp/build
ncnn/cpp/build/yolo_ncnn check-postprocess artifacts/ncnn
BENCH_SMOKE=1 bash native/bench_engine.sh
```

Para outro diretório gerado, usar `generate.py --output DIRETÓRIO` e
`cmake ... -DENGINE_GENERATED_DIR=DIRETÓRIO_ABSOLUTO`. Somente o conjunto de
operadores/atributos descrito pelo compilador é aceito; não é um runtime
ONNX geral. Entradas dinâmicas, outros modelos, batch > 1 e outras CPUs de
afinidade padrão não foram validados. Os testes TSan cobrem kernels/pool,
não um forward completo instrumentado de todas as oito imagens. A
aproximação de exp tem teste denso, não prova formal para todos os floats.

Próximos passos: concluir a sessão térmica, reavaliar NB=32 vs 64 com o
mesmo protocolo, melhorar reutilização de pesos/cache e divisão de trabalho
nas camadas pequenas antes de alterar layout. **INT8 com AVX-VNNI = PBI-098**,
com calibração e validação próprias; não foi implementado neste cartão.
