# PBI-095 — YOLOv10s com TensorRT na GPU, rodada 2

Execução: 11/11/2026. Mesmo modelo fundido end-to-end da rodada 1, batch 1,
entrada estática `1×3×640×640`, saída `1×300×6`, sem NMS externo.
Caminho: PyTorch → ONNX → `trtexec` → runner C++ com `enqueueV3`, **sem libtorch**.
As referências e os dois `.pt2` da rodada 1 foram copiados, não regenerados.

**Resultado principal, após reconstrução dos três engines:** TensorRT FP16 e
FP32 com `--noTF32` passaram nas oito entradas e no check de imagem. FP16 mediu
**2,95 ms (1,89× o eager FP16)**; FP32 noTF32, **7,87 ms (1,25× o eager FP32)**.
O FP32 padrão mediu 7,44 ms e mantém o resultado honesto **7/8 PASS; 1 FAIL**
por score em `webcam_04`. A causa foi confirmada: TF32 habilitado por padrão
no TensorRT versus referência eager em FP32 estrito. Desligá-lo numa variante
separada custa 0,43 ms / 5,8% de latência nesta sequência. Nenhuma tolerância foi
alterada; o engine padrão continua com sua política original.

## Ambiente e bibliotecas

| Componente | Versão / configuração desta rodada |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU, capacidade 8.6, 20 SMs |
| Driver | 620.70.06 |
| TensorRT / parser ONNX | 11.4.0.106, sistema; `trtexec` v110400 b106 |
| CUDA do `yolo_trt` | `/opt/cuda`, runtime 13.4; `cudaRuntimeGetVersion=13040`, `libcudart.so.13` |
| CUDA do `yolo_aoti` / eager | wheels cu128, runtime 12.8.90; `cudaRuntimeGetVersion=12080`, `libcudart.so.12` |
| cuDNN do sistema | 9.28.1, `cudnnGetVersion()=92801` |
| cuDNN das wheels Torch | 9.19.0.56, `torch.backends.cudnn.version()=91900` |
| Python / PyTorch | 3.12.13 / 2.11.0+cu128 |
| Ultralytics | 8.4.210 |
| ONNX / ONNXScript / ONNX Runtime | 1.23.2 / 0.7.2 / 1.31.1 |
| g++ | 16.2.1 20260810, C++17 |
| CMake / Ninja | 4.4.5 / 1.13.2 |

Os dois runtimes CUDA ficam em **processos separados**. O CMake do TensorRT não
procura Torch, Python ou nvcc: importa `nvinfer`, o `cudart` do `/opt/cuda` e `dl`.
O AOTI continua usando exclusivamente os headers/runtime das wheels no seu build.
A disponibilidade de cuDNN 9.28.1 no sistema não significa que o runner TRT tenha
uma dependência direta dela: ela não aparece no seu `ldd`.

Dependências adicionadas ao projeto, sem mudar versões preexistentes:

- `onnx`: formato, inspeção e `onnx.checker.check_model(..., full_check=True)`;
- `onnxscript`: necessário ao exportador ONNX moderno (`dynamo=True`) do torch 2.11;
- `onnxruntime`: executa uma vez cada ONNX no input de bus para conferir a saída.
  Usa o provider CPU **somente para checagem de exportação/diagnóstico**, nunca
  para a tabela de desempenho. Não se trata de um benchmark de CPU.
- Dependências transitivas: `onnx-ir`, `ml-dtypes`, `protobuf` e `flatbuffers`.

## Código comum e preservação da rodada 1

`native/common.h` é header-only e não inclui Torch/ATen. Compartilha leitura de
binários e de detecções eager, `Detection`, IoU e pareamento guloso, validação,
letterbox/resize, COCO, NVML opcional por `dlopen`, percentis e impressão das
métricas. `stb_image.h` foi movido com `git mv` para `native/third_party/`, sem
alteração do conteúdo ou da licença. O AOTI preserva argumentos, configurações
de TF32/cuDNN e linhas de saída.

**Refinamento da métrica:** com score ≥ 0,25 em cada lado, primeiro são casadas
as detecções pelo maior IoU restante, independentemente de classe; depois se
exigem a mesma classe e IoU ≥ 0,95. Os máximos de caixas/scores são calculados
**nos pares casados**, não na posição da linha do TopK. A contagem deve ser igual.
O máximo linha a linha de todas as 1.800 posições continua em `overall_max`,
apenas informativo. Assim uma permutação não é confundida com erro numérico.
Sem detecções em ambos os lados, os máximos filtrados são 0 e o IoU é
convencionalmente 1 — não uma medição de caixas.

Limites inalterados: FP32 caixas ≤ 0,5 px e scores ≤ 1e-3; FP16 caixas ≤ 2 px e
scores ≤ 5e-3. `detect --check` exige mesma contagem/classe e IoU ≥ 0,90 em
coordenadas originais, após letterbox C++ via stb/resize bilinear.
`aoti/export.py` e `aoti/bench_eager.py` não foram editados.

## Exportação ONNX e construção dos engines

O exportador importa `load_model` e `configure_cuda` de `aoti.export`, seleciona
a cabeça one-to-one antes da fusão e usa exportação moderna em **opset 18**,
com nomes `images` e `output0`, dimensões estáticas e pesos embutidos no ONNX.
FP16 é realmente `model.half()` com entrada half, não uma solicitação de
precisão ao builder. O wrapper restaura o dtype de saída ao dtype da entrada:
o Ultralytics converte a coluna de classe para FP32 e promove a concatenação
mesmo no modelo half. Caixas/scores já half e IDs inteiros 0–79 são exatamente
representáveis em half; essa conversão de saída não modifica seus valores.
A entrada e a saída do ONNX FP16 foram verificadas como `FLOAT16`.
Os pesos/constantes numéricos FP16 têm 174 inicializadores `FLOAT16`; o único
inicializador `FLOAT` restante é a escala `[1,1,2,2]` dos dois `Resize`.
Também há índices/formatos INT64/BOOL e casts de classe; não se afirma que
operações de forma/índice sejam half.

Saída da checagem ONNX-vs-eager, no bus, uma execução por precisão:

```text
fp32: opset 18, shape [1,300,6], onnx.checker PASS; onnx-vs-eager max abs diff 312.297333
fp16: opset 18, shape [1,300,6], onnx.checker PASS; onnx-vs-eager max abs diff 472.4375
```

Esses máximos são **globais e linha a linha**, não um teste de equivalência de
detecções: TopK pode reordenar linhas de baixa confiança. Um diagnóstico com
pareamento das cinco detecções de bus contra a referência fixa encontrou,
no ONNX Runtime: FP32 caixa máxima `0,0000610352 px`, score `2,38419e-7`,
IoU mínimo `0,9999994`; FP16 caixa `0,5 px`, score `0,000732422`, IoU `0,997264`.
Não se usou o máximo global para afrouxar as tolerâncias dos runners.

Operadores exportados: 86 `Conv`, 73 `Sigmoid`, 74 `Mul`, 19 `Concat`, 12 `Split`,
12 `Reshape`, 11 `Add`, 4 `Transpose`, 5 `Unsqueeze`, 3 `MaxPool`,
3 `GatherElements`, 2 de cada `MatMul`, `Softmax`, `Resize`, `TopK`, `Expand`
e `Sub`, além de `ReduceMax`, `Div`, `Mod`, `Less`, `Equal`, `And` e `Cast`
(3 no FP32, 6 no FP16). Atenção e TopK foram aceitos, sem custom ops/plugins
adicionados. Os três logs finais de build estão livres de linhas `[W]`/`[E]`
do parser/builder. Houve um FutureWarning Python de `TreeSpec`; os builds C++
produziram avisos não fatais de inicialização/funções não usadas no stb e,
no AOTI, avisos de headers c10, sem patch, downgrade ou supressão de checks.

`trt/build_engines.sh` chama `trtexec --stronglyTyped --skipInference`, sem
alterar otimização ou usar flag FP16 do builder. Constrói `yolov10s_fp32.plan`
e `yolov10s_fp16.plan` com a política padrão, além de
`yolov10s_fp32_notf32.plan` a partir do mesmo ONNX FP32, acrescentando somente
`--noTF32`. No TRT 11,
`--stronglyTyped` consta no help como obsoleto/no-op: redes já são fortemente
tipadas por padrão. O flag solicitado foi mantido; não há uso de APIs C++
obsoletas como binding indices ou `enqueueV2`.

### TF32 e defaults

O log do TRT 11 imprime apenas `Precision: Strongly Typed`, sem informar TF32.
Por isso o script compila/executa uma consulta pequena a `IBuilderConfig`, sem
chamar setters: **`tf32=on optimization_level=3`**. O help do `trtexec` confirma
TF32 habilitado por padrão; nenhum `--noTF32` é passado ao engine padrão.
Esse resultado é registrado no log e no sidecar `build_yolov10s_fp32.tf32`.
É uma **permissão** de usar TF32 no engine FP32, não uma alegação de que toda
camada usa TF32. Na variante separada `fp32_notf32`, o flag `--noTF32` é explícito
e o sidecar `build_yolov10s_fp32_notf32.tf32` registra `off`.
No bloco FP16, TF32 é impresso como `n/a`.

| Engine | Bytes | MiB | Build de parede (s) |
|---|---:|---:|---:|
| `yolov10s_fp32.plan` | 113.783.508 | 108,512 | 55,686113 |
| `yolov10s_fp16.plan` | 85.612.612 | 81,647 | 60,472920 |
| `yolov10s_fp32_notf32.plan` | 114.761.372 | 109,445 | 55,555977 |

O tempo inclui o processo `trtexec`, parsing, otimização, serialização e a
checagem interna de desserialização, mas não exportação, consulta de defaults
ou download. Não é um build a frio: houve um build FP32 anterior e caches podem
estar aquecidos. O nível de otimização é o default 3. Logs e sidecars ficam em
`artifacts/trt/build_<stem>.{log,seconds,tf32}`, com stems `yolov10s_fp32`,
`yolov10s_fp16` e `yolov10s_fp32_notf32`. O runner usa o stem do plano para
consultar esses arquivos, sem confundir os dois engines FP32. Os pesos estão nos `.plan`;
o runner não precisa de ONNX nem `.pt` durante inferência. Os engines são
maiores que os `.pt2`, pois também contêm código/táticas do TensorRT.

## Validação

### AOTI recompilado com o código compartilhado

Saída real; ambos os validates e os dois detects terminaram com código 0:

```text
PASS bus fp32 boxes_max=0.022919 scores_max=0.000593 dets=5/5 min_iou=0.999774 overall_max=504.086304
PASS zidane fp32 boxes_max=0.020447 scores_max=0.000076 dets=3/3 min_iou=0.999867 overall_max=564.094788
PASS webcam_00 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=543.558838
PASS webcam_01 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=575.163574
PASS webcam_02 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=572.042664
PASS webcam_03 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=572.793091
PASS webcam_04 fp32 boxes_max=0.034546 scores_max=0.000146 dets=1/1 min_iou=0.999573 overall_max=629.473755
PASS webcam_05 fp32 boxes_max=0.001068 scores_max=0.000541 dets=1/1 min_iou=0.999967 overall_max=635.628540
PASS bus fp16 boxes_max=0.500000 scores_max=0.001953 dets=5/5 min_iou=0.997326 overall_max=485.312500
PASS zidane fp16 boxes_max=0.125000 scores_max=0.000977 dets=3/3 min_iou=0.999626 overall_max=621.531250
PASS webcam_00 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=569.750000
PASS webcam_01 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=613.250000
PASS webcam_02 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=571.812500
PASS webcam_03 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=631.171875
PASS webcam_04 fp16 boxes_max=0.000000 scores_max=0.001709 dets=1/1 min_iou=1.000000 overall_max=632.812500
PASS webcam_05 fp16 boxes_max=0.000000 scores_max=0.000000 dets=1/1 min_iou=1.000000 overall_max=632.929688
PASS detect bus fp32 dets=5/5 min_iou=0.999379
PASS detect bus fp16 dets=5/5 min_iou=0.995995
```

Um teste sintético do matcher compartilhado também passou em seis situações:
linhas permutadas (diferenças zero), caixa deslocada em 10 px, classe errada,
contagem diferente, score deslocado em 0,02 e conjuntos vazios. Não foram
alteradas referências para testar esses casos.

### TensorRT

Saída real após reconstruir todos os planos com o script atualizado. FP32
padrão: **7 PASS / 1 FAIL, código 1**; FP32 noTF32 e FP16: **8 PASS / 0 FAIL,
código 0**. As linhas de validação continuam indicando o dtype; o nome do plano
abaixo identifica a variante noTF32.

`yolov10s_fp32.plan`:

```text
engine: build s 55.686 | plan MiB 108.512 | deserialize ms 29.217
tf32: on
PASS bus fp32 boxes_max=0.016205 scores_max=0.000355 dets=5/5 min_iou=0.999837 overall_max=504.091431
PASS zidane fp32 boxes_max=0.011047 scores_max=0.000381 dets=3/3 min_iou=0.999880 overall_max=621.755920
PASS webcam_00 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=569.934204
PASS webcam_01 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=575.166565
PASS webcam_02 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=569.240662
PASS webcam_03 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=601.781250
FAIL webcam_04 fp32 boxes_max=0.011932 scores_max=0.001922 dets=1/1 min_iou=0.999764 overall_max=632.405396 reason=score tolerance exceeded;
PASS webcam_05 fp32 boxes_max=0.004395 scores_max=0.000691 dets=1/1 min_iou=0.999829 overall_max=633.374573
```

`yolov10s_fp32_notf32.plan`:

```text
engine: build s 55.556 | plan MiB 109.445 | deserialize ms 29.641
tf32: off
PASS bus fp32 boxes_max=0.000031 scores_max=0.000001 dets=5/5 min_iou=0.999999 overall_max=0.000122
PASS zidane fp32 boxes_max=0.000031 scores_max=0.000000 dets=3/3 min_iou=1.000000 overall_max=0.000122
PASS webcam_00 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=16.730530
PASS webcam_01 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=0.000092
PASS webcam_02 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=560.188232
PASS webcam_03 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=0.000092
PASS webcam_04 fp32 boxes_max=0.000031 scores_max=0.000001 dets=1/1 min_iou=1.000000 overall_max=0.000244
PASS webcam_05 fp32 boxes_max=0.000000 scores_max=0.000000 dets=1/1 min_iou=1.000000 overall_max=0.000183
```

`yolov10s_fp16.plan`:

```text
engine: build s 60.473 | plan MiB 81.647 | deserialize ms 34.198
tf32: n/a
PASS bus fp16 boxes_max=0.250000 scores_max=0.001953 dets=5/5 min_iou=0.998636 overall_max=503.968750
PASS zidane fp16 boxes_max=0.250000 scores_max=0.001465 dets=3/3 min_iou=0.999252 overall_max=612.437500
PASS webcam_00 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=570.812500
PASS webcam_01 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=571.750000
PASS webcam_02 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=602.187500
PASS webcam_03 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=628.648438
PASS webcam_04 fp16 boxes_max=0.000000 scores_max=0.001465 dets=1/1 min_iou=1.000000 overall_max=625.875000
PASS webcam_05 fp16 boxes_max=0.000000 scores_max=0.001709 dets=1/1 min_iou=1.000000 overall_max=632.648438
```

Os três `detect --check` em bus terminaram com código 0, na ordem FP32 padrão,
FP32 noTF32 e FP16:

```text
PASS detect bus fp32 dets=5/5 min_iou=0.999180
PASS detect bus fp32 dets=5/5 min_iou=0.999281
PASS detect bus fp16 dets=5/5 min_iou=0.995995
```

### Causa confirmada do FAIL FP32: TF32 versus referência estrita

`webcam_04` tem uma única detecção (classe 59). Não há ambiguidade de
pareamento/reordenação nessa detecção. O score eager fixo é `0,2705571651`;
o ONNX Runtime CPU retorna `0,2705569863` (diferença `1,78814e-7`, caixas
idênticas). Na primeira construção do engine padrão, TRT retornou
aproximadamente `0,268386`, diferença `0,002171` (cerca de **0,0022**).
O erro também foi reproduzido por `trtexec --loadInputs`, descartando o runner
C++ como causa. Após a reconstrução atual, o mesmo caso continua FAIL, com
diferença `0,001922`; a seleção de táticas pode variar entre builds.

A referência eager foi gerada com TF32 desligado. Já o TensorRT permite TF32
por padrão em tensores FP32: armazenamento FP32 não garante aritmética FP32
estrita. O `.plan` original contém kernels de convolução/GEMM com nomes
`f32f32_tf32f32_f32`. O teste controlado usa o **mesmo ONNX FP32 e os mesmos
flags**, acrescentando somente `--noTF32` em um plano separado. Esse plano
passa nas oito entradas. Isso confirma **TF32 como causa da discrepância**, sem
mudar as referências ou afrouxar limites. Eager/AOTI de benchmark continuam com
cuDNN TF32 ligado e matmul TF32 desligado; isso não muda a referência estrita.

Uma checagem com mais casas decimais mediu, em todas as detecções casadas do
engine noTF32, caixa máxima **3,0517578125e-5 px (≈3e-5 px)** e score máximo
**7,45058059692e-7 (<1e-6)**. Em `webcam_04`, o score difere apenas
`5,36441802979e-7`. Em bus, as linhas completas têm `overall_max=0,000122`;
em `webcam_04`, `0,000244`. Neste rebuild, `webcam_00` e `webcam_02`, sem
nenhuma detecção ≥ 0,25, ainda reordenam linhas de baixa confiança, com
`overall_max` grande. Portanto não se generaliza a coincidência linha a linha
a todos os builds: o pareamento segue necessário e o máximo global é informativo.

O FAIL do engine padrão é **uma questão da tolerância numérica de score, não
um erro de detecção nesta amostra**: mesma contagem e classes, caixas praticamente
iguais e IoU de `webcam_04` ≈0,9998; o score permanece acima de 0,25 nos dois
lados. Isso não elimina o FAIL nem garante equivalência em um dataset maior.

## Benchmark das sete variantes

Uma única sequência, ordem eager/AOTI/TRT FP32/TRT FP32 noTF32, depois
eager/AOTI/TRT FP16. A variante noTF32 vem imediatamente após TRT FP32.
Cada variante roda em processo separado. Entrada de bus copiada para a GPU
uma vez; somente forward, 50 aquecimentos e 500 medidas; sincronização antes
do cronômetro e depois do forward. TRT usa um stream e `cudaStreamSynchronize`
com `enqueueV3`, sem CUDA Graph. Percentis com interpolação linear; throughput
é `1000/média`, não FPS de câmera. Nenhum autotuning cuDNN foi habilitado.
Eager/AOTI mantiveram `cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off`.
TRT permite TF32 no FP32 padrão e o desliga explicitamente no FP32 noTF32;
as políticas de matmul, bibliotecas e táticas não
são idênticas às dos outros backends, mesmo com o mesmo input e protocolo.

Carga exclui inicialização explícita do contexto CUDA e importação das
bibliotecas. No TRT inclui leitura do `.plan`, criação do runtime,
desserialização, contexto, buffers e endereços; o tempo de desserialização
isolado também é mostrado. Primeiro forward fica fora da média/aquecimento.
RSS é o pico do processo inteiro, não só pesos; NVML consulta o PID após o
benchmark, não o pico. Sem arquivos sidecar, o runner mostra metadados `n/a`.

| Variante | Média (ms) | p50 (ms) | p95 (ms) | it/s | GPU processo (MiB) | RSS pico (MiB) | Carga (ms) | Primeiro forward (ms) | Validação |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eager-py fp32 | 9,86 | 9,71 | 10,73 | 101,4 | 244,0 | 1296,6 | 45,42 | 212,29 | referência |
| aoti-cpp fp32 | 8,54 | 8,48 | 9,23 | 117,1 | 204,0 | 1033,3 | 27,39 | 1624,05 | 8/8 PASS |
| trt-cpp fp32 | 7,44 | 7,25 | 8,23 | 134,3 | 262,0 | 665,9 | 98,58 | 36,57 | **7/8 PASS; 1 FAIL** |
| trt-cpp fp32 notf32 | 7,87 | 7,71 | 8,69 | 127,0 | 262,0 | 659,8 | 104,60 | 33,67 | 8/8 PASS |
| eager-py fp16 | 5,59 | 5,47 | 6,42 | 178,9 | 232,0 | 1682,0 | 52,50 | 1602,70 | referência |
| aoti-cpp fp16 | 3,89 | 3,81 | 4,66 | 257,1 | 174,0 | 1207,6 | 20,03 | 1931,02 | 8/8 PASS |
| trt-cpp fp16 | 2,95 | 2,88 | 3,72 | 338,5 | 218,0 | 645,1 | 61,57 | 33,80 | 8/8 PASS |

Speedups calculados a partir das médias impressas, sempre contra eager da
mesma precisão: AOTI FP32 **1,15× / −13,4%**, AOTI FP16 **1,44× / −30,4%**;
TRT FP32 **1,33× / −24,5% (FAIL na tolerância estrita)**,
TRT FP32 noTF32 **1,25× / −20,2%**, TRT FP16 **1,89× / −47,2%**.
TRT FP16 é 1,32× o AOTI FP16 (−24,2%). Desligar TF32 aumenta a média FP32
em **0,43 ms / 5,8%**, mantendo a mesma memória NVML de 262 MiB.
Os números desta sequência de sete processos substituem a tabela anterior de seis.

Saída literal de `bash scripts/bench_all.sh`, também salva em
`artifacts/bench/round2.txt`:

```text
variant: eager-py fp32
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 9.86 | p50 9.71 | p95 10.73
throughput: 101.4 it/s
process GPU MiB (NVML): 244.0
peak RSS MiB: 1296.6
load ms: 45.42 | first run ms: 212.29
variant: aoti-cpp fp32
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 8.54 | p50 8.48 | p95 9.23
throughput: 117.1 it/s
process GPU MiB (NVML): 204.0
peak RSS MiB: 1033.3
load ms: 27.39 | first run ms: 1624.05
variant: trt-cpp fp32
settings: strongly_typed=on builder_optimization=default cudnn_benchmark=n/a
latency ms: mean 7.44 | p50 7.25 | p95 8.23
throughput: 134.3 it/s
process GPU MiB (NVML): 262.0
peak RSS MiB: 665.9
load ms: 98.58 | first run ms: 36.57
engine: build s 55.686 | plan MiB 108.512 | deserialize ms 27.555
tf32: on
variant: trt-cpp fp32 notf32
settings: strongly_typed=on builder_optimization=default cudnn_benchmark=n/a
latency ms: mean 7.87 | p50 7.71 | p95 8.69
throughput: 127.0 it/s
process GPU MiB (NVML): 262.0
peak RSS MiB: 659.8
load ms: 104.60 | first run ms: 33.67
engine: build s 55.556 | plan MiB 109.445 | deserialize ms 30.330
tf32: off
variant: eager-py fp16
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 5.59 | p50 5.47 | p95 6.42
throughput: 178.9 it/s
process GPU MiB (NVML): 232.0
peak RSS MiB: 1682.0
load ms: 52.50 | first run ms: 1602.70
variant: aoti-cpp fp16
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 3.89 | p50 3.81 | p95 4.66
throughput: 257.1 it/s
process GPU MiB (NVML): 174.0
peak RSS MiB: 1207.6
load ms: 20.03 | first run ms: 1931.02
variant: trt-cpp fp16
settings: strongly_typed=on builder_optimization=default cudnn_benchmark=n/a
latency ms: mean 2.95 | p50 2.88 | p95 3.72
throughput: 338.5 it/s
process GPU MiB (NVML): 218.0
peak RSS MiB: 645.1
load ms: 61.57 | first run ms: 33.80
engine: build s 60.473 | plan MiB 81.647 | deserialize ms 32.718
tf32: n/a
```

Antes da sequência, NVML não listou outro processo CUDA de computação; o desktop
continuou ativo. A GPU estava em 48,81 °C, SM 1732 MHz e 18,41 W; depois,
61,47 °C, SM 1957 MHz e 83,00 W. Não são amostras contínuas. Não foram travados
clocks/potência e a alimentação pela tomada não foi confirmada por software.
Boost, perfil de energia e compartilhamento térmico CPU/GPU do notebook afetam
os resultados. É uma sequência, não um estudo com ordem aleatória ou intervalos
de confiança. A ordem alternada reduz, mas não elimina, deriva térmica.

## Dependências nativas (`ldd`)

`ldd build/bin/yolo_trt` resolveu tudo, sem `not found`:

```text
libnvinfer.so.11 => /usr/lib/libnvinfer.so.11
libcudart.so.13 => /opt/cuda/lib64/libcudart.so.13
libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6
libdl.so.2, libpthread.so.0, librt.so.1, loader ELF e linux-vdso
```

**Não há libtorch, libc10, libpython ou libtorch_python.** O parser
`libnvonnxparser` é usado no build pelo `trtexec`, não pelo runner. NVML é
opcional por `dlopen`, e o driver CUDA é carregado em execução, portanto não
são dependências diretas dessa lista. `yolo_aoti` continua com Torch/c10,
`libcudart.so.12` e demais bibliotecas das wheels, sem Python; seu `ldd` foi
repetido e não apresentou bibliotecas não resolvidas. Ambos rodam diretamente,
sem Python e com `env -u LD_LIBRARY_PATH`.

## Reprodução exata

Na raiz da árvore, com as referências fixas da rodada 1 disponíveis. Não rodar
`aoti/export.py`: isso substituiria a ground truth. Nesta worktree vazia de
artefatos, o preparo foi (origem somente leitura):

```bash
mkdir -p artifacts
cp -r /home/nicolas/Projects/experimentos/compiled-cnn/artifacts/{data,ref,yolov10s_fp32.pt2,yolov10s_fp16.pt2,export_metadata.json} artifacts/
cp /home/nicolas/Projects/experimentos/compiled-cnn/yolov10s.pt .
uv sync
cmake --preset release
cmake --build --preset release
for precision in fp32 fp16; do
  env -u LD_LIBRARY_PATH build/bin/yolo_aoti validate \
    "artifacts/yolov10s_${precision}.pt2" artifacts/ref
done
uv run python trt/export_onnx.py
bash trt/build_engines.sh
cmake --preset release
cmake --build --preset release
# Não usar set -e neste loop: o FP32 padrão retorna 1 honestamente nesta rodada.
for variant in fp32 fp32_notf32 fp16; do
  env -u LD_LIBRARY_PATH build/bin/yolo_trt validate \
    "artifacts/yolov10s_${variant}.plan" artifacts/ref
  echo "validate $variant exit=$?"
done
for backend in aoti trt; do
  extension=pt2
  if [ "$backend" = trt ]; then extension=plan; fi
  env -u LD_LIBRARY_PATH "build/bin/yolo_$backend" detect \
    "artifacts/yolov10s_fp32.$extension" artifacts/data/bus.jpg \
    --check artifacts/ref/bus.eager_dets.txt
  env -u LD_LIBRARY_PATH "build/bin/yolo_$backend" detect \
    "artifacts/yolov10s_fp16.$extension" artifacts/data/bus.jpg \
    --check artifacts/ref/bus.eager_fp16_dets.txt
done
env -u LD_LIBRARY_PATH build/bin/yolo_trt detect \
  artifacts/yolov10s_fp32_notf32.plan artifacts/data/bus.jpg \
  --check artifacts/ref/bus.eager_dets.txt
bash scripts/bench_all.sh
ldd build/bin/yolo_trt
ldd build/bin/yolo_aoti
uv run python -m py_compile trt/export_onnx.py aoti/export.py aoti/bench_eager.py
bash -n trt/build_engines.sh
bash -n scripts/bench_all.sh
git diff --check
```

Diagnósticos adicionais executados, sem alterar o engine:

```bash
trtexec --loadEngine=artifacts/yolov10s_fp32.plan --dumpLayerInfo --skipInference
trtexec --loadEngine=artifacts/yolov10s_fp16.plan --dumpLayerInfo --skipInference
strings artifacts/yolov10s_fp32.plan | grep -Ei '(tf32|f32|ampere)' | head -30
trtexec --loadEngine=artifacts/yolov10s_fp32.plan \
  --loadInputs=images:artifacts/ref/webcam_04.input.bin \
  --iterations=1 --duration=0 --warmUp=0 \
  --exportOutput=artifacts/trt/webcam04_output.json
```

Os logs, outputs e artefatos grandes ficam em `artifacts/`, fora do Git. O
runner verifica nomes, dimensões, formato e dtypes dos tensores do engine,
converte float↔half no host quando necessário e usa endereços por nome.
Contexto, engine, runtime, buffers e stream têm RAII. O dtype vem do engine,
não de uma suposição baseada na extensão do arquivo.

## Conclusão e limitações

O runner nativo TensorRT sem libtorch está demonstrado. **FP16 é a variante
mais rápida validada nesta pequena amostra**: 2,95 ms / 338,5 it/s, redução de
47,2% contra eager FP16 e 24,2% contra AOTI FP16. O primeiro forward TRT também
foi menor (cerca de 34 ms), embora seja preciso pagar a construção prévia do
engine. O RSS caiu, mas **não houve ganho de memória de GPU contra AOTI**:
TRT usou 218 versus 174 MiB em FP16 e 262 versus 204 MiB em FP32. O `.plan`
também é maior que o `.pt2`.

A variante **FP32 noTF32** oferece equivalência estrita nas oito entradas:
caixas ≈3e-5 px e scores <1e-6, com média de 7,87 ms, ainda 20,2% menor que
o eager FP32. O preço de desligar TF32 é **0,43 ms / 5,8%** sobre os 7,44 ms
do engine padrão nesta sequência. O FP32 padrão continua honestamente FAIL
no limite de score: a diferença original de cerca de 0,0022 em uma detecção
(0,001922 no rebuild atual), com mesmas classes/contagens e caixas quase iguais,
é **uma questão de tolerância numérica, não um erro de detecção observado**.
Isso não autoriza chamá-lo de substituto numericamente validado com os limites
fixados, mas torna explícito o compromisso entre precisão e latência.
O AOTI preserva o resultado anterior de 16/16 PASS com o código comum.

Oito entradas, quatro sem detecções, não substituem avaliação de acurácia em
dataset. As medições dependem dos clocks/térmica do notebook, das bibliotecas
e das políticas de precisão. Os arquivos **`.plan` são específicos da GPU e
da versão do TensorRT**; devem ser reconstruídos no destino compatível, não
tratados como formato portátil. Não foram avaliados CPU, INT8, Torch-TensorRT,
webcam C++ ou portabilidade entre GPUs.
