# PBI-095 — YOLOv10s com TensorRT na GPU, rodada 2

Execução: 11/11/2026. Mesmo modelo fundido end-to-end da rodada 1, batch 1,
entrada estática `1×3×640×640`, saída `1×300×6`, sem NMS externo.
Caminho: PyTorch → ONNX → `trtexec` → runner C++ com `enqueueV3`, **sem libtorch**.
As referências e os dois `.pt2` da rodada 1 foram copiados, não regenerados.

**Resultado principal:** TensorRT FP16 passou nas oito entradas e no check de
imagem, com média de **2,86 ms (1,90× o eager FP16)**. TensorRT FP32 mediu
7,19 ms, mas **não passou na validação completa: 7/8 PASS**, por diferença de
score em `webcam_04`. O ganho FP32 não deve ser apresentado como equivalência
numérica validada. Nenhuma tolerância nem política de TF32 foi alterada para
obter aprovação.

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
adicionados. Ambos os logs finais de build estão livres de linhas `[W]`/`[E]`
do parser/builder. Houve um FutureWarning Python de `TreeSpec`; os builds C++
produziram avisos não fatais de inicialização/funções não usadas no stb e,
no AOTI, avisos de headers c10, sem patch, downgrade ou supressão de checks.

`trt/build_engines.sh` chama `trtexec --stronglyTyped --skipInference`, sem
alterar otimização, TF32 ou usar flag FP16 do builder. No TRT 11,
`--stronglyTyped` consta no help como obsoleto/no-op: redes já são fortemente
tipadas por padrão. O flag solicitado foi mantido; não há uso de APIs C++
obsoletas como binding indices ou `enqueueV2`.

### TF32 e defaults

O log do TRT 11 imprime apenas `Precision: Strongly Typed`, sem informar TF32.
Por isso o script compila/executa uma consulta pequena a `IBuilderConfig`, sem
chamar setters: **`tf32=on optimization_level=3`**. O help do `trtexec` confirma
TF32 habilitado por padrão; nenhum `--noTF32` foi passado. Esse resultado é
registrado no log e no sidecar `build_fp32.tf32`. É uma **permissão** de usar
TF32 no engine FP32, não uma alegação de que toda camada usa TF32.
No bloco FP16, TF32 é impresso como `n/a`.

| Engine | Bytes | MiB | Build de parede (s) |
|---|---:|---:|---:|
| `yolov10s_fp32.plan` | 113.915.748 | 108,639 | 54,983453 |
| `yolov10s_fp16.plan` | 85.463.012 | 81,504 | 59,289675 |

O tempo inclui o processo `trtexec`, parsing, otimização, serialização e a
checagem interna de desserialização, mas não exportação, consulta de defaults
ou download. Não é um build a frio: houve um build FP32 anterior e caches podem
estar aquecidos. O nível de otimização é o default 3. Logs e sidecars ficam em
`artifacts/trt/build_{fp32,fp16}.{log,seconds,tf32}`. Os pesos estão nos `.plan`;
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

Saída real dos validates. FP32 terminou com **código 1**, FP16 com código 0:

```text
engine: build s 54.983 | plan MiB 108.639 | deserialize ms 28.974
tf32: on
PASS bus fp32 boxes_max=0.009888 scores_max=0.000142 dets=5/5 min_iou=0.999881 overall_max=462.113831
PASS zidane fp32 boxes_max=0.008057 scores_max=0.000055 dets=3/3 min_iou=0.999938 overall_max=621.755005
PASS webcam_00 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=575.507141
PASS webcam_01 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=575.166870
PASS webcam_02 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=602.808105
PASS webcam_03 fp32 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=601.781616
FAIL webcam_04 fp32 boxes_max=0.011169 scores_max=0.002171 dets=1/1 min_iou=0.999752 overall_max=632.405029 reason=score tolerance exceeded;
PASS webcam_05 fp32 boxes_max=0.004639 scores_max=0.000582 dets=1/1 min_iou=0.999841 overall_max=635.620300
engine: build s 59.290 | plan MiB 81.504 | deserialize ms 30.949
tf32: n/a
PASS bus fp16 boxes_max=0.250000 scores_max=0.001953 dets=5/5 min_iou=0.998636 overall_max=503.968750
PASS zidane fp16 boxes_max=0.250000 scores_max=0.002441 dets=3/3 min_iou=0.999252 overall_max=627.843750
PASS webcam_00 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=569.812500
PASS webcam_01 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=574.687500
PASS webcam_02 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=602.187500
PASS webcam_03 fp16 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=628.625000
PASS webcam_04 fp16 boxes_max=0.000000 scores_max=0.001465 dets=1/1 min_iou=1.000000 overall_max=626.054688
PASS webcam_05 fp16 boxes_max=0.000000 scores_max=0.000000 dets=1/1 min_iou=1.000000 overall_max=632.648438
```

Os dois `detect --check` em bus terminaram com código 0:

```text
PASS detect bus fp32 dets=5/5 min_iou=0.999206
PASS detect bus fp16 dets=5/5 min_iou=0.995995
```

### Investigação do FAIL FP32, sem alterar a política do builder

`webcam_04` tem uma única detecção (classe 59). Não há ambiguidade de
pareamento/reordenação nessa detecção. O score eager fixo é `0,2705571651`;
o ONNX Runtime CPU retorna `0,2705569863` (diferença `1,78814e-7`, caixas
idênticas). O mesmo engine TRT, carregado também por `trtexec --loadInputs`,
retorna aproximadamente `0,268386`, reproduzindo a diferença `0,002171`.
Assim a discrepância surge no caminho de execução TensorRT, não na geração
ONNX, no pré-processamento de imagem ou apenas no runner C++.

A configuração permite TF32; a inspeção do `.plan` com `strings` encontra
kernels de convolução e GEMM como
`sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f32f32_tf32f32_f32_...` e
`sm86_xmma_gemm_f32f32_tf32f32_f32_...`, além de kernels F32.
Isso é compatível com arredondamento TF32/táticas/fusões diferentes do eager
estrito da referência. **Não prova qual operação isolada causou o erro**.
`--dumpLayerInfo` nos engines finais retorna apenas nomes: a verbosidade padrão
de build é `kLAYER_NAMES_ONLY`, sem formatos/táticas por camada. Não houve
rebuild desligando TF32 nem alteração do engine aprovado para mascarar o FAIL.
Uma atribuição causal exata por camada permanece em aberto; o limite original
continua valendo e o resultado FP32 permanece FAIL.

## Benchmark das seis variantes

Uma única sequência, ordem eager/AOTI/TRT FP32, depois eager/AOTI/TRT FP16.
Cada variante roda em processo separado. Entrada de bus copiada para a GPU
uma vez; somente forward, 50 aquecimentos e 500 medidas; sincronização antes
do cronômetro e depois do forward. TRT usa um stream e `cudaStreamSynchronize`
com `enqueueV3`, sem CUDA Graph. Percentis com interpolação linear; throughput
é `1000/média`, não FPS de câmera. Nenhum autotuning cuDNN foi habilitado.
Eager/AOTI mantiveram `cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off`.
TRT permite TF32 no FP32; as políticas de matmul, bibliotecas e táticas não
são idênticas às dos outros backends, mesmo com o mesmo input e protocolo.

Carga exclui inicialização explícita do contexto CUDA e importação das
bibliotecas. No TRT inclui leitura do `.plan`, criação do runtime,
desserialização, contexto, buffers e endereços; o tempo de desserialização
isolado também é mostrado. Primeiro forward fica fora da média/aquecimento.
RSS é o pico do processo inteiro, não só pesos; NVML consulta o PID após o
benchmark, não o pico. Sem arquivos sidecar, o runner mostra metadados `n/a`.

| Variante | Média (ms) | p50 (ms) | p95 (ms) | it/s | GPU processo (MiB) | RSS pico (MiB) | Carga (ms) | Primeiro forward (ms) | Validação |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eager-py fp32 | 9,72 | 9,72 | 9,80 | 102,8 | 244,0 | 1306,7 | 59,72 | 198,84 | referência |
| aoti-cpp fp32 | 8,28 | 8,28 | 8,32 | 120,8 | 204,0 | 1034,2 | 14,50 | 1555,13 | 8/8 PASS |
| trt-cpp fp32 | 7,19 | 7,19 | 7,24 | 139,0 | 262,0 | 667,1 | 97,32 | 34,71 | **7/8 PASS; 1 FAIL** |
| eager-py fp16 | 5,43 | 5,42 | 5,49 | 184,1 | 232,0 | 1681,1 | 50,52 | 1554,37 | referência |
| aoti-cpp fp16 | 3,81 | 3,79 | 3,88 | 262,3 | 174,0 | 1207,5 | 8,91 | 1850,09 | 8/8 PASS |
| trt-cpp fp16 | 2,86 | 2,85 | 2,90 | 350,2 | 218,0 | 645,2 | 58,12 | 33,30 | 8/8 PASS |

Speedups calculados a partir das médias impressas, sempre contra eager da
mesma precisão: AOTI FP32 **1,17× / −14,8%**, AOTI FP16 **1,43× / −29,8%**;
TRT FP32 **1,35× / −26,0% (não validado)**, TRT FP16 **1,90× / −47,3%**.
TRT FP16 é 1,33× o AOTI FP16 (−24,9%).

Saída literal de `bash native/bench_all.sh`, também salva em
`artifacts/bench/round2.txt`:

```text
variant: eager-py fp32
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 9.72 | p50 9.72 | p95 9.80
throughput: 102.8 it/s
process GPU MiB (NVML): 244.0
peak RSS MiB: 1306.7
load ms: 59.72 | first run ms: 198.84
variant: aoti-cpp fp32
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 8.28 | p50 8.28 | p95 8.32
throughput: 120.8 it/s
process GPU MiB (NVML): 204.0
peak RSS MiB: 1034.2
load ms: 14.50 | first run ms: 1555.13
variant: trt-cpp fp32
settings: strongly_typed=on builder_optimization=default cudnn_benchmark=n/a
latency ms: mean 7.19 | p50 7.19 | p95 7.24
throughput: 139.0 it/s
process GPU MiB (NVML): 262.0
peak RSS MiB: 667.1
load ms: 97.32 | first run ms: 34.71
engine: build s 54.983 | plan MiB 108.639 | deserialize ms 27.993
tf32: on
variant: eager-py fp16
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 5.43 | p50 5.42 | p95 5.49
throughput: 184.1 it/s
process GPU MiB (NVML): 232.0
peak RSS MiB: 1681.1
load ms: 50.52 | first run ms: 1554.37
variant: aoti-cpp fp16
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 3.81 | p50 3.79 | p95 3.88
throughput: 262.3 it/s
process GPU MiB (NVML): 174.0
peak RSS MiB: 1207.5
load ms: 8.91 | first run ms: 1850.09
variant: trt-cpp fp16
settings: strongly_typed=on builder_optimization=default cudnn_benchmark=n/a
latency ms: mean 2.86 | p50 2.85 | p95 2.90
throughput: 350.2 it/s
process GPU MiB (NVML): 218.0
peak RSS MiB: 645.2
load ms: 58.12 | first run ms: 33.30
engine: build s 59.290 | plan MiB 81.504 | deserialize ms 30.696
tf32: n/a
```

Antes da sequência, NVML não listou outro processo CUDA de computação; o desktop
continuou ativo. A GPU estava em 37,03 °C, SM 210 MHz e 3,82 W; depois,
54,97 °C, SM 1972 MHz e 82,28 W. Não são amostras contínuas. Não foram travados
clocks/potência e a alimentação pela tomada não foi confirmada por software.
Boost, perfil de energia e compartilhamento térmico CPU/GPU do notebook afetam
os resultados. É uma sequência, não um estudo com ordem aleatória ou intervalos
de confiança. A ordem alternada reduz, mas não elimina, deriva térmica.

## Dependências nativas (`ldd`)

`ldd trt/cpp/build/yolo_trt` resolveu tudo, sem `not found`:

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
cmake -S aoti/cpp -B aoti/cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(uv run python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
cmake --build aoti/cpp/build
for precision in fp32 fp16; do
  env -u LD_LIBRARY_PATH aoti/cpp/build/yolo_aoti validate \
    "artifacts/yolov10s_${precision}.pt2" artifacts/ref
done
uv run python trt/export_onnx.py
bash trt/build_engines.sh
cmake -S trt/cpp -B trt/cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build trt/cpp/build
# Não usar set -e neste loop: o FP32 retorna 1 honestamente nesta rodada.
for precision in fp32 fp16; do
  env -u LD_LIBRARY_PATH trt/cpp/build/yolo_trt validate \
    "artifacts/yolov10s_${precision}.plan" artifacts/ref
  echo "validate $precision exit=$?"
done
for backend in aoti trt; do
  extension=pt2
  if [ "$backend" = trt ]; then extension=plan; fi
  env -u LD_LIBRARY_PATH "$backend/cpp/build/yolo_$backend" detect \
    "artifacts/yolov10s_fp32.$extension" artifacts/data/bus.jpg \
    --check artifacts/ref/bus.eager_dets.txt
  env -u LD_LIBRARY_PATH "$backend/cpp/build/yolo_$backend" detect \
    "artifacts/yolov10s_fp16.$extension" artifacts/data/bus.jpg \
    --check artifacts/ref/bus.eager_fp16_dets.txt
done
bash native/bench_all.sh
ldd trt/cpp/build/yolo_trt
ldd aoti/cpp/build/yolo_aoti
uv run python -m py_compile trt/export_onnx.py aoti/export.py aoti/bench_eager.py
bash -n trt/build_engines.sh
bash -n native/bench_all.sh
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
mais rápida validada nesta pequena amostra**: 2,86 ms / 350,2 it/s, redução de
47,3% contra eager FP16 e 24,9% contra AOTI FP16. O primeiro forward TRT também
foi menor (cerca de 33 ms), embora seja preciso pagar a construção prévia do
engine. O RSS caiu, mas **não houve ganho de memória de GPU contra AOTI**:
TRT usou 218 versus 174 MiB em FP16 e 262 versus 204 MiB em FP32. O `.plan`
também é maior que o `.pt2`.

FP32 apresentou ganho de latência, porém **a equivalência numérica exigida não
foi atingida**. Não é correto promovê-lo como substituto validado com estes
limites; a diferença de score continua aberta para investigação por camada.
O AOTI preserva 16/16 PASS depois da extração do código comum.

Oito entradas, quatro sem detecções, não substituem avaliação de acurácia em
dataset. As medições dependem dos clocks/térmica do notebook, das bibliotecas
e das políticas de precisão. Os arquivos **`.plan` são específicos da GPU e
da versão do TensorRT**; devem ser reconstruídos no destino compatível, não
tratados como formato portátil. Não foram avaliados CPU, INT8, Torch-TensorRT,
webcam C++ ou portabilidade entre GPUs.
