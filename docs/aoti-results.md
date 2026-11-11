# PBI-095 — YOLOv10s com AOTInductor na GPU

Execução: 11/11/2026. Escopo: batch 1, entrada estática `1×3×640×640`, GPU,
YOLOv10s fundida, cabeça end-to-end, sem NMS externo. Python exporta e gera a
referência; o consumidor de inferência é um processo C++ sem Python.

## Ambiente

| Componente | Versão / configuração |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU, capacidade 8.6 |
| CPU | Intel Core i5-13450HX (ambiente informado) |
| Driver NVIDIA | 615.71.09; `nvidia-smi` anuncia CUDA UMD 13.4 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128; CUDA do build: 12.8 |
| Runtime CUDA efetivamente usado | wheel `nvidia-cuda-runtime-cu12` 12.8.90; `cudaRuntimeGetVersion = 12080` |
| cuDNN | wheel 9.19.0.56; `torch.backends.cudnn.version() = 91900` |
| Triton | 3.6.0, incluindo headers CUDA 12.8 e `crt/` |
| Ultralytics / OpenCV Python | 8.4.210 / 5.0.0 |
| Compilador C++ | g++ 16.2.1 20260810, C++17, ABI C++11 do libstdc++ habilitada |
| CMake / Ninja | 4.4.3 / 1.13.2 |
| Toolkit do sistema | CUDA 13.4 em `/opt/cuda`, **não usado no build final** |

Nos benchmarks, ambos os consumidores configuram explicitamente os padrões do
PyTorch: **TF32 do cuDNN ligado, TF32 de matmul desligado**. Há variantes com
`cudnn.benchmark=False` e `True`, sempre alinhadas entre Python e C++.
A exportação e a geração de referência continuam estritas, com ambos os TF32
desligados. Não foi instalado pacote de sistema.

## Compatibilidades e decisões de implementação

1. **Cabeça do modelo:** neste Ultralytics, o checkpoint antigo carrega com
   `head.end2end=False`, apesar de conter os pesos one-to-one. Apenas definir
   `export=True` produziu um tensor `[1,84,8400]`, incompatível com o experimento.
   O exportador define `head.end2end=True` **antes de fundir** o modelo e então
   `head.export=True`. O forward foi executado e a saída foi verificada:
   exatamente um tensor `[1,300,6]` em ambas as precisões. Os campos são
   `(x1,y1,x2,y2,score,classe)` em pixels da imagem com letterbox.
2. **Exportação:** `torch.export.export` seguido de
   `torch._inductor.aoti_compile_and_package`; não se usa a API obsoleta
   `torch._export.aot_compile`. O FP16 usa modelo e entrada em half. Convoluções,
   atenção, decodificação e topk exportaram e compilaram sem custom ops adicionais.
3. **CUDA do AOTInductor:** a primeira investigação revelou que o compilador
   escolhia automaticamente os headers e caminhos de link de `/opt/cuda`.
   Esses pacotes preliminares foram substituídos. O exportador agora monta
   `artifacts/cuda-12.8/` com links para headers/runtime das wheels, aponta
   `CUDA_HOME` e o valor em cache de `torch.utils.cpp_extension.CUDA_HOME` para
   esse diretório e não chama nvcc. A wheel de runtime omite `crt/`, obtido do
   Triton 3.6.0, também CUDA 12.8.
4. **Dependência adicional necessária:** `nvidia-cuda-cccl-cu12==12.8.90` fornece
   `nv/target`, incluído por `cuda_fp16.h`. Sem ela, a tentativa de compilar com
   apenas runtime + Triton falhou com `fatal error: nv/target: No such file or
   directory`. A wheel de CCCL fornece apenas os headers de que o build precisa;
   nenhuma versão de dependência preexistente foi alterada. Os diretórios de
   headers da CCCL também integram o `CUDA_HOME` local.
5. **CMake sem toolkit:** o runner tem apenas `.cpp`. Em vez de `find_package(Torch)`,
   que habilitaria a linguagem CUDA e procuraria nvcc desnecessariamente, o CMake
   localiza a wheel pelo `CMAKE_PREFIX_PATH`/`Torch_DIR`, importa explicitamente as
   bibliotecas Torch e usa headers/runtime CUDA 12.8 das wheels. Isso evita
   completamente o toolkit 13.4. O ABI `_GLIBCXX_USE_CXX11_ABI=1` foi conferido com
   `torch._C._GLIBCXX_USE_CXX11_ABI` nesta wheel.
6. **GCC 16:** ambos os builds funcionaram sem patch de headers, downgrade,
   `-fpermissive` ou flags para contornar erros. Houve avisos não fatais em
   `stb_image.h` (inicializadores) e nos headers de `c10::InferenceMode`
   (`-Wmaybe-uninitialized`), além de avisos Python de depreciação de TreeSpec e
   de TF32 desabilitado. O aviso de poucos SMs para `max_autotune_gemm` também
   apareceu; não foi forçado autotuning adicional.
7. **Imagem e NVML:** `stb_image.h` foi baixado do repositório oficial, mantendo
   sua licença. SHA-256:
   `fc644f6b893f5225d59670ebb3e5384771a58cfbfa5e5ca27fd85cd8da220056`.
   O NVML é opcional em C++ via `dlopen("libnvidia-ml.so.1")`, com init/shutdown,
   enumeração de dispositivos e `nvmlDeviceGetComputeRunningProcesses_v3`.
   A memória corresponde ao PID do próprio processo, não à ocupação total da GPU;
   falhas de consulta retornam `n/a`. Python usa `pynvml` com o mesmo critério.

## Artefatos e pesos

| Pacote | Bytes | MB decimais | MiB | Exportação + compilação (s) |
|---|---:|---:|---:|---:|
| `yolov10s_fp32.pt2` | 31.216.228 | 31,216 | 29,770 | 13,043 |
| `yolov10s_fp16.pt2` | 16.725.668 | 16,726 | 15,951 | 9,926 |

Os tempos são os da última exportação bem-sucedida: incluem `torch.export`,
compilação e empacotamento, com sincronização CUDA, mas não incluem download,
carga/fusão, captura ou geração da referência. **Não são tempos de build a frio**:
houve tentativas prévias e caches parcialmente aquecidos. Os caches do build
final ficam em `artifacts/inductor_cache/` e `artifacts/triton_cache/`, evitando
usar o `/tmp` em tmpfs como armazenamento de grandes caches.

**Os pesos estão embutidos.** Cada `.pt2` foi inspecionado como ZIP (67 entradas),
contendo wrapper C++, metadados, cubins e uma biblioteca `.wrapper.so`. O comando
de link registrado no wrapper inclui um `.weights.o`; no ELF resultante a seção
`.lrodata` tem 29.096.664 bytes em FP32 e 14.548.344 bytes em FP16. As bibliotecas
medem, respectivamente, 29.716.784 e 15.168.448 bytes. Não há arquivo externo de
pesos necessário à inferência, nem é necessário o checkpoint `.pt` no C++.
O loader extrai temporariamente os componentes nativos do ZIP; isso não é
compilação ou execução Python em tempo de inferência.

A inspeção dos wrappers finais confirmou a presença de
`artifacts/cuda-12.8/include` e a ausência de `/opt/cuda`. O `ldd` das duas
bibliotecas extraídas também não contém `/opt/cuda` nem `libpython`.

## Entradas e referência

- `bus.jpg` e `zidane.jpg`: copiados de `ultralytics/assets/` instalado.
- Seis frames capturados **uma vez** em `/dev/video0`, V4L2, MJPG, 1280×720,
  após descartar 30 frames de exposição. As novas exportações reutilizaram os
  seis arquivos; não houve falha de captura nesta execução.
- Os arquivos de imagem e todos os artefatos estão fora do Git.
- Python: `LetterBox(640×640, auto=False)`, proporção preservada, padding
  centralizado 114, BGR→RGB, CHW, divisão por 255 e float32.
- `manifest.txt` lista oito nomes. Cada `.input.bin` tem 4.915.200 bytes; cada
  `.eager_fp32.bin` e `.eager_fp16.bin` tem 7.200 bytes. Todos são float32 bruto
  little-endian sem cabeçalho; a saída eager FP16 é convertida para FP32 antes
  de salvar. O runner converte a entrada FP32 para half no pacote FP16.
- `<nome>.eager_dets.txt` contém detecções eager FP32 em coordenadas originais,
  no formato `classe score x1 y1 x2 y2`. A referência equivalente FP16 é
  `<nome>.eager_fp16_dets.txt`.

## Validação AOTI contra eager da mesma precisão

Limites mantidos: FP32 ≤ 0,5 px e ≤ 1e-3 no score; FP16 ≤ 2 px e ≤ 5e-3.
As diferenças elementares de caixas/scores só consideram linhas em que pelo
menos um lado tem score ≥ 0,25. A contagem das detecções deve coincidir; o
pareamento guloso pelo maior IoU restante exige mesma classe e IoU ≥ 0,95.
`overall_max` considera todas as linhas e colunas, sendo apenas informativo.

Saída real (ambos os comandos terminaram com código 0):

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
```

Resultado: **16/16 PASS**. As grandes diferenças globais vêm da reordenação de
linhas de baixa confiança no topk e não são usadas como tolerância. Quatro
frames não têm detecções ≥ 0,25; nesses casos os máximos filtrados são zero e o
IoU mínimo é convencionalmente 1, ou seja, não representam uma medição de caixas.
Essa amostra pequena não substitui uma avaliação de acurácia em um dataset.

### Caminho de imagem inteiramente em C++

`stb_image` fornece RGB; o runner faz resize bilinear com convenção de meio pixel,
arredonda para uint8, centraliza padding 114 e normaliza. Não depende de OpenCV.
O `--check` compara a contagem e as classes, com pareamento por IoU ≥ 0,90 **depois**
de mapear ambas as saídas para a imagem original, compensando padding e escala,
e limitando as caixas às dimensões da imagem.

```text
class=5 name=bus score=0.962745 box=16.303162,229.552216,802.608459,736.329224
class=0 name=person score=0.919670 box=50.274540,397.086914,245.864700,905.035767
class=0 name=person score=0.901850 box=223.567337,405.536987,345.911560,859.226074
class=0 name=person score=0.897928 box=669.442932,393.076355,809.160400,881.144775
class=0 name=person score=0.449087 box=0.039049,555.800476,77.343384,871.204834
PASS detect bus fp32 dets=5/5 min_iou=0.999379
PASS detect bus fp16 dets=5/5 min_iou=0.995995
```

O FP16 foi comparado ao arquivo de detecções eager FP16. Ambos terminaram com
código 0. Testes negativos adicionais, com cópias temporárias da referência,
confirmaram código 1 para caixa alterada em 10 px, score alterado em 0,02,
classe incorreta, arquivo binário truncado e contagem incorreta em `detect`:
**5/5 comportamentos esperados**. Um teste com câmera simulada falhando após o
primeiro frame salvo também confirmou que o exportador descarta a captura
parcial e retorna somente os dois assets. As referências originais não foram alteradas.

## Benchmark

Protocolo idêntico: entrada `bus.input.bin` já na GPU, só forward, primeira
execução separada, 50 aquecimentos e 500 iterações medidas, sincronização CUDA
antes do cronômetro e depois do forward, relógio de parede (`perf_counter` /
`steady_clock`). Percentis com interpolação linear. Throughput é `1000 / média`,
não FPS de um pipeline de câmera. Cada uma das oito variantes roda em processo
separado para não contaminar o pico de RSS, o cache de autotuning ou a memória
NVML com outra variante. A ordem foi eager/AOTI FP32 sem autotuning, eager/AOTI
FP32 com autotuning, eager/AOTI FP16 sem autotuning e eager/AOTI FP16 com autotuning,
sem rodar outras medições entre esses oito processos.

**Correção da primeira rodada:** o eager herdava `cudnn.allow_tf32=False` do
exportador, enquanto o runner C++ deixava TF32 do cuDNN ligado por padrão.
Isso inflava o ganho FP32 publicado anteriormente. Os números abaixo substituem
os anteriores: Python agora define `cudnn.allow_tf32=True`,
`cuda.matmul.allow_tf32=False` e o valor de `cudnn.benchmark`; C++ usa
`at::globalContext().setAllowTF32CuDNN(true)`, `setAllowTF32CuBLAS(false)`
e `setBenchmarkCuDNN(...)`. Cada bloco imprime os valores efetivos. O exportador
não foi alterado e a validação contra suas referências estritas foi repetida:
**8 PASS / 0 FAIL em FP32 e 8 PASS / 0 FAIL em FP16**, com os mesmos valores
por amostra apresentados acima.

O resultado anterior de aproximadamente 7 ms no script de webcam com
Ultralytics `predict` usava letterbox retangular com entrada 640×384, enquanto
este benchmark usa sempre 640×640.

**AOTI chama cuDNN, não é um grafo com todas as convoluções em Triton.** A inspeção
de cada wrapper encontrou 86 chamadas `aoti_torch_cuda_convolution`; um perfil
diagnóstico separado, fora das medições, registrou `aten::cudnn_convolution`
84 vezes por forward FP32 e 86 por forward FP16. Ativar o autotuning no AOTI
reduziu a média de 8,33 para 8,10 ms em FP32 e de 3,81 para 3,51 ms em FP16.

A carga eager inclui checkpoint, seleção da cabeça, fusão e transferência para
CUDA/precisão solicitada. A carga AOTI inclui extração e carga do pacote. Importação
de Python/bibliotecas e inicialização explícita do contexto CUDA ficam fora dessa
métrica; ela **não** é o tempo de inicialização completo do processo. A primeira
execução não está na média e pode incluir inicializações tardias de CUDA/cuDNN.
RSS é o pico durante a vida inteira de cada processo (`ru_maxrss`), não memória
somente dos pesos. Memória NVML é uma leitura após o benchmark, não seu pico.

| Variante | Média (ms) | p50 (ms) | p95 (ms) | it/s | GPU processo (MiB) | RSS pico (MiB) | Carga (ms) | Primeiro forward (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eager-py fp32 | 9,75 | 9,74 | 9,82 | 102,6 | 244,0 | 1329,6 | 55,87 | 280,53 |
| aoti-cpp fp32 | 8,33 | 8,33 | 8,35 | 120,1 | 204,0 | 1051,1 | 15,80 | 1570,62 |
| eager-py fp32 cudnn-bench | 9,21 | 9,21 | 9,29 | 108,5 | 250,0 | 1478,6 | 58,98 | 15318,63 |
| aoti-cpp fp32 cudnn-bench | 8,10 | 8,10 | 8,15 | 123,4 | 198,0 | 1329,0 | 15,86 | 18530,63 |
| eager-py fp16 | 5,45 | 5,44 | 5,50 | 183,6 | 232,0 | 1752,5 | 62,66 | 1645,43 |
| aoti-cpp fp16 | 3,81 | 3,80 | 3,88 | 262,3 | 174,0 | 1217,5 | 17,99 | 1931,68 |
| eager-py fp16 cudnn-bench | 5,22 | 5,22 | 5,28 | 191,4 | 234,0 | 1941,1 | 53,49 | 13906,42 |
| aoti-cpp fp16 cudnn-bench | 3,51 | 3,50 | 3,57 | 284,9 | 176,0 | 1632,0 | 9,79 | 11350,84 |

Saída literal das oito variantes, na ordem da execução:

```text
variant: eager-py fp32
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 9.75 | p50 9.74 | p95 9.82
throughput: 102.6 it/s
process GPU MiB (NVML): 244.0
peak RSS MiB: 1329.6
load ms: 55.87 | first run ms: 280.53
variant: aoti-cpp fp32
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 8.33 | p50 8.33 | p95 8.35
throughput: 120.1 it/s
process GPU MiB (NVML): 204.0
peak RSS MiB: 1051.1
load ms: 15.80 | first run ms: 1570.62
variant: eager-py fp32 cudnn-bench
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=on
latency ms: mean 9.21 | p50 9.21 | p95 9.29
throughput: 108.5 it/s
process GPU MiB (NVML): 250.0
peak RSS MiB: 1478.6
load ms: 58.98 | first run ms: 15318.63
variant: aoti-cpp fp32 cudnn-bench
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=on
latency ms: mean 8.10 | p50 8.10 | p95 8.15
throughput: 123.4 it/s
process GPU MiB (NVML): 198.0
peak RSS MiB: 1329.0
load ms: 15.86 | first run ms: 18530.63
variant: eager-py fp16
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 5.45 | p50 5.44 | p95 5.50
throughput: 183.6 it/s
process GPU MiB (NVML): 232.0
peak RSS MiB: 1752.5
load ms: 62.66 | first run ms: 1645.43
variant: aoti-cpp fp16
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=off
latency ms: mean 3.81 | p50 3.80 | p95 3.88
throughput: 262.3 it/s
process GPU MiB (NVML): 174.0
peak RSS MiB: 1217.5
load ms: 17.99 | first run ms: 1931.68
variant: eager-py fp16 cudnn-bench
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=on
latency ms: mean 5.22 | p50 5.22 | p95 5.28
throughput: 191.4 it/s
process GPU MiB (NVML): 234.0
peak RSS MiB: 1941.1
load ms: 53.49 | first run ms: 13906.42
variant: aoti-cpp fp16 cudnn-bench
settings: cudnn_tf32=on matmul_tf32=off cudnn_benchmark=on
latency ms: mean 3.51 | p50 3.50 | p95 3.57
throughput: 284.9 it/s
process GPU MiB (NVML): 176.0
peak RSS MiB: 1632.0
load ms: 9.79 | first run ms: 11350.84
```

Não havia outro processo CUDA de computação antes desses oito benchmarks; o
desktop GNOME continuava ativo. Não foram travados clocks ou potência. A GPU
estava em 39 °C, SM 210 MHz e 3,94 W antes da sequência, e em 58 °C,
SM 1965 MHz e 84,60 W logo depois. Não são amostras contínuas durante cada iteração. A alimentação pela tomada
não foi confirmada independentemente pelo software; temperatura, perfil de
energia, boost e compartilhamento térmico CPU/GPU de notebook limitam a
comparabilidade. Estes números são uma rodada sequencial, não um estudo com
ordem aleatória e intervalos de confiança. Repetir na tomada para comparar
outras máquinas e sessões.

## Dependências nativas (`ldd`)

O `ldd aoti/cpp/build/yolo_aoti` resolveu todas as bibliotecas, sem `not found`:

- Wheel Torch: `libtorch.so`, `libtorch_cpu.so`, `libtorch_cuda.so`, `libc10.so`,
  `libc10_cuda.so`, `libtorch_nvshmem.so` e `libgomp.so.1`.
- Wheels NVIDIA em `.venv/lib/python3.12/site-packages/nvidia/`: `libcudart.so.12`,
  `libcupti.so.12`, `libcusparse.so.12`, `libcufft.so.11`, `libcufile.so.0`,
  `libcusparseLt.so.0`, `libnccl.so.2`, `libcurand.so.10`, `libcublas.so.12`,
  `libcublasLt.so.12`, `libcudnn.so.9`, `libnvshmem_host.so.3`, `libnvJitLink.so.12`.
- Sistema: glibc, libstdc++, libgcc_s, libm, pthread, rt, dl, util e o loader ELF.
- O driver `libcuda.so.1` é necessário em execução; NVML é carregado via `dlopen`
  e por isso não aparece como dependência direta no `ldd` do executável.
- **Não há `libpython` nem `libtorch_python`**. O binário foi executado diretamente
  com `env -u LD_LIBRARY_PATH`, sem `uv run` ou interpretador Python.

O CMake grava `RUNPATH` absoluto para Torch e bibliotecas NVIDIA na `.venv`.
Não é preciso exportar `LD_LIBRARY_PATH` nesta árvore. Isso não torna a distribuição
independente das bibliotecas: ao mover a árvore/venv, é necessário reconfigurar e
recompilar o runner, ou preparar um layout/RPATH de distribuição equivalente.
Os nomes `yolov10s_fp32.pt2`/`yolov10s_fp16.pt2` são o contrato de seleção de
precisão do runner; ele rejeita nomes desconhecidos em vez de adivinhar o dtype.
O benchmark encontra a referência em `ref/bus.input.bin` ao lado do pacote.

## Reprodução

Executar na raiz do projeto, na mesma máquina, com GPU disponível:

```bash
uv sync
uv run python aoti/export.py --capture-webcam
cmake -S aoti/cpp -B aoti/cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(uv run python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
cmake --build aoti/cpp/build

for precision in fp32 fp16; do
  env -u LD_LIBRARY_PATH aoti/cpp/build/yolo_aoti validate \
    "artifacts/yolov10s_${precision}.pt2" artifacts/ref
done
# Oito processos, alternando eager/AOTI para distribuir a deriva térmica.
for precision in fp32 fp16; do
  for tuning in off on; do
    flags=()
    if [ "$tuning" = on ]; then flags+=(--cudnn-benchmark); fi
    uv run python aoti/bench_eager.py --precision "$precision" "${flags[@]}"
    env -u LD_LIBRARY_PATH aoti/cpp/build/yolo_aoti bench \
      "artifacts/yolov10s_${precision}.pt2" "${flags[@]}"
  done
done
env -u LD_LIBRARY_PATH aoti/cpp/build/yolo_aoti detect \
  artifacts/yolov10s_fp32.pt2 artifacts/data/bus.jpg \
  --check artifacts/ref/bus.eager_dets.txt
env -u LD_LIBRARY_PATH aoti/cpp/build/yolo_aoti detect \
  artifacts/yolov10s_fp16.pt2 artifacts/data/bus.jpg \
  --check artifacts/ref/bus.eager_fp16_dets.txt
ldd aoti/cpp/build/yolo_aoti
```

O exportador baixa `yolov10s.pt` se necessário. Para capturar novamente, remover
apenas `artifacts/data/webcam_*.jpg` antes da exportação; por padrão a captura
completa anterior é preservada. Se a webcam falhar, o script descarta uma captura
parcial, continua com as duas imagens dos assets e informa isso. `artifacts/export_metadata.json` registra
versões, amostras, tempo, tamanho e entradas ZIP de cada pacote.

Inspeção dos pacotes e seções ELF, sem depender de unzip:

```bash
uv run python - <<'PY'
import json, zipfile
from pathlib import Path
m = json.loads(Path('artifacts/export_metadata.json').read_text())
for precision in ('fp32', 'fp16'):
    p = m['packages'][precision]
    print(precision, p['bytes'], p['compile_seconds'])
    with zipfile.ZipFile(f'artifacts/yolov10s_{precision}.pt2') as z:
        print('\n'.join(z.namelist()))
        so = next(n for n in z.namelist() if n.endswith('.so'))
        cpp = next(n for n in z.namelist() if n.endswith('.wrapper.cpp'))
        assert '/opt/cuda' not in z.read(cpp).decode()
        dest = Path('artifacts/package_inspection') / precision / 'model.so'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(so))
PY
for precision in fp32 fp16; do
  readelf -SW "artifacts/package_inspection/$precision/model.so" | grep lrodata
done
uv run python -m py_compile aoti/export.py aoti/bench_eager.py
git diff --check
```

## Conclusão

A inferência nativa sem Python foi demonstrada, com **16/16 validações PASS**
novamente contra a referência estrita; os checks de imagem da rodada anterior
continuam registrados acima. Com TF32/cuDNN finalmente alinhados, sem autotuning,
AOTI reduziu a latência média em **14,6% no FP32 (1,17×)** e **30,1% no FP16
(1,43×)** em relação ao eager da mesma precisão. Portanto, o ganho FP32 é menor
que os 22,3% divulgados na comparação anterior incorreta.

Com autotuning habilitado nos dois lados, a redução foi de **12,1% no FP32
(1,14×)** e **32,8% no FP16 (1,49×)**. A variante mais rápida foi AOTI FP16 com
cuDNN benchmark: **3,51 ms / 284,9 it/s**. Sem autotuning, a memória de GPU do
processo caiu 40 MiB em FP32 e 58 MiB em FP16; o pico de RSS também foi menor.

O ganho é de regime estacionário, não uma promessa de inicialização rápida:
ativar autotuning elevou o primeiro forward para aproximadamente **11–19 s**
e aumentou o pico de RSS. Sem autotuning, o primeiro forward AOTI ainda foi
mais lento que o eager nas duas precisões. Os pacotes contêm os pesos, mas
continuam dependendo de libtorch, bibliotecas CUDA e driver compatíveis.
A rodada corrigida confirma um ganho mensurável neste hardware, sujeito às
limitações térmicas de notebook; não avalia CPU, INT8, outros compiladores,
webcam em C++ ou acurácia ampla do modelo.
