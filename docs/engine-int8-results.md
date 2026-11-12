# PBI-098 — YOLOv10s INT8 com AVX-VNNI no engine próprio

## Resultado

**Acurácia aprovada:** 8/8 no critério INT8 da rodada 3 e mAP50–95 de
**49,0700**, contra **49,3272** do engine FP32: queda de **0,2572 ponto**
(limite: 1,0). As **70 convoluções densas** usam INT8; nenhuma foi devolvida
a FP32. As 16 depthwise, atenção, operações não Conv da cabeça e ativações
entre operações continuam em FP32.

**Desempenho: meta ainda não atingida.** Medida rápida final em seis threads:
**22,84 ms** com perfil, contra **50,19 ms** do FP32 sem perfil, aproximadamente
**2,20×** de aceleração nesta sequência. Não chegou a <20 ms nem aos 14,91 ms
do OpenVINO INT8 da rodada 3. O host estava ruidoso, com navegador/chat abertos;
essas medidas não substituem uma sessão térmica comum.

O caminho FP32 preserva os cinco arquivos do gerador sem calibração byte a
byte. Os logs completos de validação FP32, incluindo todos os números, são
idênticos antes/depois nos três modos: referência/1 thread, otimizado/1 thread
e otimizado/6 threads. Não foram alteradas referências nem tolerâncias.

## Construção efetiva

- Um executável, `--precision fp32|int8`, padrão FP32. O gerador com
  `--calibration` acrescenta pesos INT8 e dispatch por `model.ctx.int8`;
  sem a opção, emite exatamente a saída do PBI-097. Um modelo gerado sem
  calibração rejeita a seleção INT8, em vez de executar FP32 silenciosamente.
- Pesos simétricos por canal: `scale_w=max(abs(w))/127`, guarda de escala 1
  para canal inteiramente zero, `rint` half-even e saturação em [−127,127].
  Layout offline `[ceil(OC/6),ceil(K/4),6,4]`, com pesos de cauda zerados.
  Também são emitidas as escalas por canal, os produtos `scale_a*scale_w`
  e o bias corrigido `bias − scale_a*scale_w*zp*sum(qw)`.
- Ativações assimétricas por tensor: incluem zero no intervalo, escala
  `(hi−lo)/255`, `zp=round(-lo/scale)` em [0,255]. A arena continua FP32.
  O pack faz multiplicação pela escala inversa, clamp, conversão por
  `_mm256_cvtps_epi32` no MXCSR padrão (nearest-even) e soma do zero-point.
  Não há uma arena de ativações quantizadas nem im2col da imagem inteira.
- Painéis de ativação `[micro,ceil(K/4),16,4]`, lendo diretamente NCHW FP32.
  Padding espacial, colunas não utilizadas e cauda K recebem **zero-point**,
  não o inteiro zero. O caminho 1×1 contíguo quantiza quatro canais por vez;
  o geral usa loads contíguos ou deinterleave em stride 2, com gather
  mascarado nas bordas e nos painéis que atravessam linhas. Coordenadas
  são calculadas fora do laço de canais e o descritor é privado por thread.
- Microkernel **6×16**, doze acumuladores int32, K inteiro, u8×s8 sem saturação
  intermediária, usando `_mm256_dpbusd_avx_epi32`. O assembly contém
  **`{vex} vpdpbusd ... %ymm`**; não usa AVX-512. Os acumuladores permanecem
  em registradores no laço K. A referência inteira usa laços sobre os mesmos
  inteiros/layout, conferidos exatamente antes do epílogo.
- Epílogo: int32 → FP32, multiplicação pelo produto de escalas, soma do bias
  corrigido, a mesma `silu8` do FP32 e residual, saída FP32. Só o arquivo
  INT8 recebe `-mavxvnni -ffp-contract=off`; a compilação FP32 continua
  `-O3 -march=x86-64-v3 -fno-math-errno`. O teste do epílogo também desabilita
  contração para comparar a mesma sequência de operações arredondadas.
- Mantido o grid 2-D existente: **NB=16**, chunks de OC múltiplos de seis,
  pelo menos 24 jobs quando a forma permite e limite de cache do gerador.
  Chunks de canais ainda podem repetir o pack. O scratch é privado por
  worker; descritores thread-local evitam alocação por job no pack geral.
- CPUID folha 7/subfolha 1/EAX bit 4, junto da detecção AVX2/estado YMM do
  compilador: INT8 sem AVX-VNNI falha com erro claro. O executável já exige
  a plataforma x86-64-v3 para o FP32, como antes.
- `serve` lê registros little-endian FP32 `[3,640,640]` e devolve `[300,6]`
  por pipes binários, sem texto em stdout. EOF limpo termina; registro
  truncado ou entrada não finita é erro. O cliente Python devolve `(1,300,6)`.
  São seis threads fixadas em `0,2,4,6,8,10` na validação e no mAP.

O blob duplo tem **66.001.348 bytes** (aproximadamente 62,94 MiB), contra
58.570.624 bytes do FP32. Não é compressão do executável: os pesos FP32
originais e empacotados permanecem disponíveis para os dois modos FP32.

## Calibração própria e escolha do método

`engine/quant/calibrate.py` expõe como saídas ONNX Runtime CPU os tensores
**originais ONNX que entram em cada Conv densa**, inclusive os resultados
Mul/SiLU que depois viram aliases no compilador. São 70 Conv e 68 tensores
distintos. Não se calibra por nomes de raízes da arena após as fusões.

Foram usadas exatamente as **300 imagens COCO val2017 de ranks 501–800 por
image id**, via `coco_images(500,800)` e `read_image` existentes. A primeira
passagem obtém extrema exatos; a segunda acumula histogramas uniformes de
65.536 bins por tensor, sem amostragem aleatória. O JSON guarda min/max,
contagem e extremos p99,99/p99,999: caudas iguais, arredondadas para fora
às bordas dos bins. Assim, a precisão dos percentis é limitada ao bin,
mas o min/max não é aproximado. IDs, protocolo e mapeamento Conv→tensor
estão no JSON, com chaves ordenadas e sem timestamps.

Duas execuções completas produziram o mesmo SHA-256 de `calibration.json`:

```text
d220de24e61a3e2addcdf381f8c2aa24518cd6ac8b5eb155535e70f511c71d14
```

A ordem de seleção foi min/max → p99,999. O primeiro método passou mAP, mas
falhou nos scores de bus e zidane; só então foi testado o percentil seguinte.

| Tentativa | Método | Densas mantidas FP32 | Critério INT8 | mAP FP32 | mAP INT8 | Queda (pontos) |
|---|---|---|---:|---:|---:|---:|
| 1 | minmax | nenhuma | 6/8 | 49,3272 | 48,8044 | 0,5228 |
| 2 — selecionada | p99999 | nenhuma | **8/8** | 49,3272 | **49,0700** | **0,2572** |

No min/max, `scores_max` foi **0,060936** em bus e **0,077282** em zidane,
acima de 0,05. Em p99999 caiu para **0,044237** e **0,004369**. Como o segundo
método passou os dois critérios, **p9999 e fallbacks por camada não foram
tentados**. Não se consultaram as 500 imagens de avaliação para calibração,
nem se mudou a seleção de imagens ou o matcher.

## Validação das oito imagens

Chamadas reais a `cpu/common.compare(..., int8=True)` contra eager FP32;
pares por geometria com IoU ≥0,80, classe igual, diferença de score ≤0,05,
não pareados permitidos somente abaixo de 0,30. O FP32 também passa 8/8
com o modo estrito desse mesmo helper Python.

| Amostra | Resultado | boxes_max (px) | scores_max | dets real/ref | min_iou | overall_max |
|---|---|---:|---:|---:|---:|---:|
| bus | PASS | 2,958618 | 0,044237 | 5/5 | 0,934963 | 504,152802 |
| zidane | PASS | 1,908142 | 0,004369 | 3/3 | 0,977165 | 628,822998 |
| webcam_00 | PASS | 0,000000 | 0,000000 | 0/0 | 1,000000 | 573,377075 |
| webcam_01 | PASS | 0,000000 | 0,000000 | 0/0 | 1,000000 | 587,067261 |
| webcam_02 | PASS | 0,000000 | 0,000000 | 0/0 | 1,000000 | 571,780457 |
| webcam_03 | PASS | 0,000000 | 0,000000 | 0/0 | 1,000000 | 632,576965 |
| webcam_04 | PASS | 1,000122 | 0,001675 | 1/1 | 0,975801 | 626,742371 |
| webcam_05 | PASS | 0,539215 | 0,007756 | 1/1 | 0,983653 | 633,618896 |

`overall_max` compara linhas TopK por índice e inclui permutações de baixa
confiança; não é o critério de aprovação. Linhas vazias não medem qualidade
de localização; a avaliação COCO complementa essas oito referências.

`detect artifacts/data/bus.jpg --precision int8 --check ...` também passa:
**5/5, IoU mínimo 0,949275**. Esse checker compartilhado usa IoU ≥0,90 e
letterbox C++; não é o critério Python INT8. Por compatibilidade, sua linha
final ainda escreve o rótulo legado `fp32`; o CLI imprime antes dela
`precision: engine int8` e explica o rótulo, sem alterar o helper nativo
compartilhado fora do escopo.

## mAP — protocolo da rodada 3

As mesmas 500 imagens de ranks **1–500** por image id, score ≥0,001, até
300 saídas, categorias COCO ordenadas, conversão de caixas por
`cpu.eval_map.original_boxes`, `pycocotools.COCOeval` com os mesmos maxDets
padrão. Não há NMS. Os resultados e predições são gravados somente em
`artifacts/engine/int8/`, sem sobrescrever a rodada 3.

| Implementação | mAP50–95 | Queda contra respectivo FP32 |
|---|---:|---:|
| Engine FP32 | 49,3272 | — |
| **Engine INT8 p99999** | **49,0700** | **0,2572 — PASS** |
| Eager FP32 — rodada 3, referência histórica | 49,33 | — |
| OpenVINO FP32 — rodada 3, referência histórica | 49,33 | — |
| OpenVINO INT8 — rodada 3, referência histórica | 49,02 | 0,31 |

O mAP do engine FP32 coincide com o eager arredondado a duas casas; isso
não significa igualdade bit a bit de todo o tensor bruto.

## Latências rápidas e perfil

Host **ruidoso**, navegador/chat abertos. Sempre
`taskset -c 0,2,4,6,8,10`, 6 threads, 20 aquecimentos + **200** medidas finais.
Tempo inclui forward e pós-processamento top-300, mas não leitura de imagem,
letterbox ou IPC Python. INT8 com `--profile`; FP32 sem instrumentação.

| Variante | Média ms | p50 ms | p95 ms | p5 ms |
|---|---:|---:|---:|---:|
| **Engine INT8 6t** | **22,84** | **22,29** | **24,87** | **21,20** |
| Engine FP32 6t | 50,19 | 49,72 | 53,79 | 47,96 |
| OpenVINO INT8 — rodada 3, host ocioso, outra sessão | 14,91 | — | — | — |

INT8: carga 3,05 ms, primeira inferência 24,18 ms, RSS máximo 76,6 MiB.
FP32: carga 2,52 ms, primeira 50,35 ms, RSS 107,4 MiB. Páginas de pesos não
tocadas no modo INT8 ajudam a explicar o RSS menor apesar do blob maior.
Os 950,09 “GFLOPS efetivos” impressos em INT8 são trabalho FP32 equivalente
por segundo, **não** FLOPs executados nem 118,76% de utilização física FP32.

### Perfil INT8 final — seis threads

Modelo **22,18 ms**, pós **0,66 ms**. As primeiras dez linhas são tempos de
parede por chamada gerada. As duas últimas são **subcategorias sobrepostas**:
soma de durações dos jobs dividida por seis, com contadores privados por
worker e agregação após a barreira. Não se somam novamente ao modelo; não
incluem da mesma maneira dispatch, desbalanceamento e espera.

| Categoria | ms/inferência | % do tempo de modelo |
|---|---:|---:|
| conv3x3 | 7,46 | 33,63% |
| conv3x3s2 | 3,48 | 15,67% |
| conv1x1 | 7,59 | 34,21% |
| dwconv FP32 | 2,00 | 9,00% |
| maxpool | 0,14 | 0,65% |
| resize | 0,18 | 0,83% |
| attention | 0,60 | 2,70% |
| concat/copy | 0,14 | 0,63% |
| head não Conv | 0,55 | 2,50% |
| other | 0,01 | 0,03% |
| ↳ INT8 pack/quantize, normalizado por workers | **5,47** | **24,68%** |
| ↳ INT8 GEMM/epílogo, normalizado por workers | **11,08** | **49,95%** |

### O que ajudou e o que foi descartado

Experimentos cumulativos, **20 + 100** medidas com perfil, exceto o ponto
inicial/final com 200. Host sem isolamento térmico: não atribuir toda variação
à alteração de código. Todos usam afinidade de seis P-cores.

| Experimento | Média ms | Pack/quantize ms | GEMM/epílogo ms | Decisão |
|---|---:|---:|---:|---|
| Implementação inicial, NB=16, 200 medidas | 25,39 | 7,64 | 10,96 | Base |
| Interleave incremental em registradores + descritor thread-local | 23,97 | 7,27 | 10,76 | Mantido |
| NB=32 + retune de chunks com pesos de 1 byte | 26,05 | 7,59 | 10,90 | Descartado |
| NB=64 + retune de chunks com pesos de 1 byte | 29,28 | 8,71 | 11,03 | Descartado |
| NB=128 + retune de chunks com pesos de 1 byte | 30,81 | 11,10 | 10,60 | Descartado |
| Volta NB=16; pack direto 1×1 com quatro canais | 23,24 | 6,61 | 10,87 | Mantido |
| Gather mascarado no caminho de borda/linha quebrada | 21,92 | 5,22 | 10,90 | Mantido |
| **Final, 200 medidas** | **22,84** | **5,47** | **11,08** | Resultado reportado |

Tiles maiores aumentaram a repetição de pack com o particionamento de OC,
sem ganho de GEMM demonstrado. Voltou-se ao grid original NB=16. Não houve
fallback FP32 para acelerar a validação nem ajuste por imagens individuais.
O ganho de pack é visível, mas GEMM/epílogo e operações FP32 restantes ainda
impedem <20 ms nesta implementação/medida. Não foram tentados layout global
blocando ativações quantizadas, AVX-512 ou mudanças de precisão das operações
não Conv — todos fora do escopo.

## Verificação e artefatos

Logs em `artifacts/engine/int8/` (fora do Git):

- `calibration-run{1,2}.txt`, `calibration-sha1.txt`,
  `calibration-determinism.txt`: duas calibrações completas, hash igual.
- `baseline-generated/`, `fp32-regenerated/`, `baseline-sha.txt`:
  cinco arquivos FP32 idênticos sem `--calibration`. Duas gerações
  calibradas também têm os mesmos cinco hashes.
- `before-{ref,opt1,opt6}.txt`, `after-*.txt`, `diff-*.txt`, e repetição
  final `final-{ref,opt1,opt6}.txt`, `final-diff-*.txt`:
  8/8 estrito nos três modos, todos os diffs vazios.
- `final-build.txt`, `final-ctest.txt`: Release limpo, sem warnings de engine,
  **5/5 CTest**. Testes de pack com half-even, clamp, zero-point não zero,
  bordas, stride 2, K%4, OC%6, caudas de coluna, K grande com produtos que
  saturariam `pmaddubsw`, igualdade int32 exata e epílogo FP32 em tolerância
  relativa 1e-5. O residual do teste evita cancelamento em zero na comparação
  puramente relativa; os testes FP32 existentes cobrem SiLU e extremos.
- `generator-tests.txt`: **16/16**, incluindo golden SHA-256 FP32,
  determinismo INT8, input após SiLU, escalas/correção, canais zero,
  half-even de pesos, fallback explícito, nomes inválidos e depthwise FP32.
- `protocol-tests.txt`: **4/4**, EOF limpo/truncado, precisão inválida,
  redirecionamento da validação INT8, múltiplos registros binários e
  FP32 estrito pelo cliente.
- `final-tsan.txt`: **5/5** no build TSan, inclusive pack/GEMM e contadores
  de perfil em seis threads; sem relatos de data race. Não é um forward
  completo de YOLO instrumentado.
- `attempt{1,2}-{validate,map}.txt`, `attempt{1,2}-map.json`,
  `final-validate.txt`, `final-map.txt`, `map-summary.json`:
  seleção e aceitação COCO; `validate-engine-*.json` com as oito amostras.
- `final-detect.txt`: detecções INT8 de bus e PASS 5/5, IoU 0,949275.
- `final-bench-{int8,fp32}-6t.txt`: 200 iterações, perfil e p5.
- `before-ldd.txt`, `final-ldd.txt`: dependências idênticas, somente vdso,
  libstdc++, libm, libgcc_s, libc e ld-linux. Sem runtime de inferência,
  OpenMP ou BLAS no executável.
- `smoke.txt`, `smoke-exit.txt`: **exit 0**, quatro variantes aceitas sem
  retry. Resumo em `artifacts/engine/bench/engine-runs/smoke/engine-summary.txt`.
  O engine usa apenas 1+5 iterações neste teste funcional; os `nan%` de CPU
  externa nas execuções curtas indicam falta de amostras suficientes do monitor,
  não uma afirmação de host ocioso. Os números do smoke não substituem as 200
  medidas finais nem preenchem o benchmark térmico abaixo.

Reprodução da configuração selecionada, sem exportadores antigos nem NNCF:

```bash
uv run python engine/quant/calibrate.py --help
uv run python engine/quant/calibrate.py
uv run python engine/gen/generate.py \
  --calibration artifacts/engine/int8/calibration.json --int8-method p99999
rm -rf engine/build
cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build
ctest --test-dir engine/build --output-on-failure
uv run python engine/tests/test_generator.py
uv run python engine/tests/test_engine_client.py
engine/build/yolo_engine validate artifacts/ref --impl ref --threads 1
engine/build/yolo_engine validate artifacts/ref --impl opt --threads 1
taskset -c 0,2,4,6,8,10 engine/build/yolo_engine validate artifacts/ref --threads 6
uv run python engine/validate_int8.py
uv run python engine/eval_map.py
engine/build/yolo_engine detect artifacts/data/bus.jpg --precision int8 \
  --check artifacts/ref/bus.eager_dets.txt
taskset -c 0,2,4,6,8,10 engine/build/yolo_engine bench \
  --precision int8 --threads 6 --iters 200 --profile
taskset -c 0,2,4,6,8,10 engine/build/yolo_engine bench \
  --precision fp32 --threads 6 --iters 200
ldd engine/build/yolo_engine
cmake -S engine -B engine/build-tsan -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENGINE_TSAN=ON
cmake --build engine/build-tsan --target test_kernels test_int8 test_thread_pool
ctest --test-dir engine/build-tsan --output-on-failure
BENCH_SMOKE=1 BENCH_CONTENDED=1 bash native/bench_engine.sh
```

## Benchmark final (controle térmico)

**PLACEHOLDER — A PREENCHER PELO COORDENADOR. Não foi executada a sessão
completa de três rodadas nesta implementação.**

O runner agora alterna quatro variantes: `engine-fp32-6t`,
`engine-int8-6t`, `ov-cpu-fp32`, `ov-cpu-int8`; remove o engine de uma thread.
Mantém cooldown de 60 s, três rodadas com início rotativo, até cinco retries,
`BENCH_CONTENDED`, p5/margem e resumo. OpenVINO mantém exatamente 20+200
iterações, seis P-cores e os parâmetros de `cpu/bench.py`, mesmo no smoke.
Todas as novas saídas ficam sob `artifacts/engine/bench/`, não em artefatos
históricos de outras rodadas. O smoke usa uma rodada, cooldown 5 s e 1+5
iterações somente para as variantes do engine; não é resultado térmico.

## Limitações e próximos pontos de medição

A meta de latência segue aberta. Não há comparação nova controlada com o
OpenVINO nem promessa de <20 ms em host ocioso; até o p5 final foi 21,20 ms.
O perfil atribui cerca de metade do modelo ao GEMM/epílogo e um quarto ao
pack/quantize; as categorias de trabalho normalizado não substituem tempos
de parede por camada. Há repetição de pack entre chunks e ativações FP32
entre operações, por decisão de projeto. Não se afirma inferência sem heap.

Percentis são estimados por histogramas; nenhuma dependência nova foi
adicionada. A checagem CPUID foi executada na máquina com AVX-VNNI, mas a
mensagem em CPU realmente sem o recurso não foi validada por emulação.
O protocolo binário é síncrono, batch 1, entrada fixa e não é um serviço de
rede. O checkpoint, referências, seleção COCO e artefatos da rodada 3
permanecem imutáveis.
