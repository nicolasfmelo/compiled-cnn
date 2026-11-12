# PBI-095 — YOLOv10s com ncnn e Vulkan, rodada 4

Execução: 11–12/11/2026. Design aprovado em
[`2026-11-11-ncnn-vulkan-design.md`](superpowers/specs/2026-11-11-ncnn-vulkan-design.md).
Mesmo checkpoint fundido one-to-one das rodadas anteriores, batch 1, entrada
estática `1×3×640×640`. As oito referências eager foram **copiadas, não
regeneradas**. Caminho: PyTorch → TorchScript → pnnx → ncnn C++, sem Python,
Torch ou CUDA no runner.

**Acurácia:** NVIDIA FP32/FP16, Intel FP32 e CPU FP32: **8/8 PASS** cada.
Intel FP16 com aritmética half: **6 PASS / 2 FAIL**, por score, sem afrouxar
limites. Os cinco `detect --check` de bus passam. A seleção top-k em C++
produz **bit-match dos 300 registros** do pós-processamento Python quando
recebe exatamente os mesmos valores brutos de bus em FP32 na CPU.

**Desempenho controlado (mediana de três rodadas):** NVIDIA FP16 resident
**9,69 ms**, NVIDIA FP32 resident **18,55 ms**, CPU FP32 **56,89 ms**, incluindo
pós-processamento externo. A iGPU Intel ficou em uma única rodada, por decisão
do usuário após medir aproximadamente **230 ms FP16 / 433 ms FP32**.

## Ambiente

| Componente | Versão/configuração |
|---|---|
| ncnn do sistema | 1.0.20261020, pacote CachyOS, Vulkan/INT8/BF16 habilitados |
| pnnx | 20261020, wheel PyPI fixada no `uv.lock` |
| Vulkan loader/instância | 1.4.363 |
| NVIDIA | NVIDIA GeForce RTX 3050 6GB Laptop GPU, Vulkan 1.4.357 |
| Driver NVIDIA | 620.70.06 |
| Intel integrada | Intel(R) Graphics (RPL-S), Vulkan 1.4.354 |
| Driver Intel | Intel open-source Mesa driver, Mesa 26.2.6-arch3.1 |
| Cooperative matrix, consulta ncnn | NVIDIA: **sim**; Intel: **não** |
| Formatos cooperative matrix NVIDIA no log ncnn | FP16 16×16×16 / 16×8×16 / 16×8×8; também INT8 e BF16 disponíveis |
| CPU | i5-13450HX; seis threads, P-cores `0,2,4,6,8,10` via taskset |
| Python / Torch / Ultralytics | 3.12.13 / 2.11.0+cu128 / 8.4.210 |
| Compilador | g++ 16.2.1, C++17, build Release com CMake/Ninja |

A única dependência Python acrescentada é **pnnx**, para conversão offline;
nenhuma versão preexistente foi alterada. Não foi instalada a binding Python
ncnn. O runner usa `find_package(ncnn)` em `/usr/lib/cmake/ncnn` e vincula
`ncnn` e `dl`; dependências transitivas incluem OpenMP, glslang e SPIRV-Tools.
O `ldd` resolveu todas as bibliotecas: **sem libtorch, libc10, libpython,
libcudart ou TensorRT**. Vulkan é carregado pelo ncnn em execução; não precisa
aparecer como dependência direta no `ldd`. NVML é opcional por `dlopen`.

O runner enumera os dispositivos e seleciona pelo nome (`NVIDIA` ou `Intel`),
não pelo índice: nesta execução Intel era 0 e NVIDIA era 1. Falha explicitamente
se o nome solicitado não existir. Ambos expõem FP16 storage/arithmetic;
somente a NVIDIA reporta cooperative matrix ao **próprio ncnn**, além das
extensões Vulkan indicadas no design. `use_cooperative_matrix=1` é a permissão
na NVIDIA; o uso efetivo em FP16 foi confirmado separadamente abaixo. Não se
afirma que toda camada executa nos tensor cores.

## Conversão e ponto de corte

`ncnn/export_ncnn.py` reutiliza `aoti.export.load_model('fp32')` e aplica
imediatamente `.cpu().float()`. Assim preserva a seleção da cabeça one-to-one
antes da fusão; a passagem transitória por CUDA ocorre só no helper de carga,
não na inferência CPU/ncnn. O exportador prepara o cache de anchors estáticos
antes do trace: sem isso a verificação do TorchScript encontra grafos diferentes
na primeira e na segunda chamada. `check_trace` permanece habilitado.

A tentativa de exportar o grafo completo chegou ao fim do pnnx, mas deixou
camadas **`torch.topk`, `torch.gather` e `Tensor.to` não convertidas** no `.param`.
São as duas seleções TopK, gathers por elemento e conversão da coluna de
classe do pós-processamento. Portanto um exit zero do conversor não bastava
para declarar suporte ao grafo completo. A atenção **PSA foi convertida**, sem
plugin nem implementação externa. Logs: `artifacts/ncnn-full-export.log` e
`artifacts/ncnn-export.log`.

O corte está **na entrada de `Detect.postprocess`, após o `permute(0,2,1)`**:
saída ncnn `[8400,84]` (batch 1 implícito), com caixas **já decodificadas xyxy**
e 80 scores **já submetidos a sigmoid**, da cabeça one-to-one. DFL, anchors,
escalas/strides, atenção e convoluções continuam dentro do ncnn. Não se trata
de logits antes de DFL, nem da cabeça one-to-many.

O `.param` final tem 252 camadas e 297 blobs, entrada `in0`, saída `out0`.
O exportador rejeita tipos de camada residuais contendo `.`. Pesos são salvos
em **FP32 (`pnnx.export(..., fp16=False)`)**, para não contaminar a variante
FP32 com compressão half. São 21.239 bytes de `.param` + 29.096.984 bytes de
`.bin`, **27,769 MiB** no total; o mesmo par serve aos cinco dispositivos/precisões.
Os intermediários TorchScript/pnnx e fixtures ficam em `artifacts/ncnn/`, fora
do Git, sem modificar `artifacts/ref/`.

### Pós-processamento C++

`ncnn/cpp/postprocess.h` reproduz as duas etapas de
`Detect.get_topk_index`, com `groups=1` (não se usa o formato `engine`):

1. máximo sobre as 80 classes de cada anchor; seleção dos 300 melhores anchors;
2. seleção dos 300 melhores scores entre os `300×80` elementos desses anchors;
3. caixas do anchor escolhido, score e classe `índice % 80`, saída `[300,6]`.

Um anchor pode aparecer mais de uma vez com classes diferentes. Não há NMS,
nem filtro de confiança nessa etapa. O filtro ≥ 0,25 é o da validação/detecção
compartilhada. Para reproduzir inclusive a ordem do empate observado em bus,
a implementação usa a mesma escolha de algoritmos do TopK CPU do Torch 2.11:
`nth_element` + `sort` na primeira seleção e `partial_sort` na segunda,
comparando somente scores. Um desempate por índice era semanticamente válido,
mas trocava duas linhas de baixa confiança em bus; foi retirado para o teste
bit a bit solicitado.

```text
PASS postprocess bus bit-match (all 300 rows)
```

Esse teste usa `bus.raw.bin` e `bus.postprocess.bin` gerados pelo Python **do
mesmo forward CPU**; não afirma igualdade bit a bit entre redes em backends
ou precisões diferentes. `torch.topk` não garante ordem portátil de empates;
CPU/GPU e versões de biblioteca padrão podem ordenar empates de outra forma.
Por isso `overall_max` linha a linha continua apenas informativo.

## Runner, opções e medição

CLI, na raiz do projeto:

```bash
build/bin/yolo_ncnn validate PARAM BIN REF_DIR --device nvidia --precision fp32
build/bin/yolo_ncnn bench PARAM BIN --device intel --precision fp16 --mode resident
build/bin/yolo_ncnn detect PARAM BIN IMAGE --device cpu --precision fp32 --check DETS_TXT
build/bin/yolo_ncnn check-postprocess artifacts/ncnn
```

`--device`: `nvidia|intel|cpu`; `--precision`: `fp32|fp16`; `--mode`:
`resident|host`. Defaults: NVIDIA, FP32, resident; CPU aceita somente FP32 e
normaliza o modo para host. Validate/detect também aceitam `--mode`: foram
conferidos os dois caminhos Vulkan. `--check` usa a referência eager da mesma
precisão. As linhas de validação/detect são exatamente as de `native/common.h`;
a linha `variant:` anterior identifica `ncnn-vk-nvidia fp16`, por exemplo.

- FP32: Vulkan habilitado nas GPUs; fp16 packed/storage/arithmetic desligados,
  BF16 storage/packed desligados.
- FP16: fp16 packed/storage/arithmetic ligados, BF16 desligado.
- Cooperative matrix ligada na NVIDIA, desligada na Intel/CPU. Packing,
  Winograd e SGEMM mantêm os defaults ligados; seis threads em todas as variantes.
- CPU: `use_vulkan_compute=0`, sem criar instância Vulkan. A afinidade externa
  é aplicada pelo script, não embutida no binário.
- Flags **somente diagnósticas**: `--storage-only`, `--no-winograd`, `--no-sgemm`,
  `--no-coop`. Não são selecionadas silenciosamente para fazer testes passarem.

**Resident:** upload de entrada uma vez para `VkMat`, fora das iterações;
cada iteração cria um extractor, extrai `VkMat` e espera `submit_and_wait`.
O cronômetro `model mean ms` termina com essa espera, com saída ainda na GPU.
Depois, `post-process mean ms` inclui **download de todos os 8400×84 valores**,
unpack/cast e top-k C++. Todos os scores são necessários à seleção exata.
Logo, apesar da entrada residente, o **total não é um forward inteiramente
residente**: inclui a transferência necessária ao pós-processamento externo.

**Host:** `Mat` de entrada → `Mat` de saída por chamada; o tempo do modelo
inclui upload/download; pós-processamento inclui unpack/cast e top-k. Ambos
reportam total de parede, média/p50/p95, throughput `1000/média`, e médias de
modelo/pós-processamento separadas. O total também inclui pequenos custos de
criação/destruição do extractor e buffers; não precisa ser exatamente a soma
dos dois componentes arredondados. Não entram leitura da imagem, letterbox,
desenho ou câmera nas iterações medidas.

GPU: primeira inferência separada, 50 aquecimentos + 500 medidas; CPU: 20 +
200. Cada forward é síncrono. Carga inclui leitura do modelo e criação dos
pipelines/compilação de shaders em `load_model`; descoberta da instância Vulkan
fica fora. **Não se fornece nem persiste um pipeline cache em disco do ncnn**;
caches internos do processo/driver podem existir, logo não se afirma carga
completamente fria. `first run ms` separa custos diferidos. RSS é o pico do
processo, NVML é amostra após as medidas, apenas NVIDIA (não pico); Intel e
CPU imprimem `n/a`. O printer local só adapta esse escopo do NVML e a divisão
dos tempos, reutilizando percentis, validação, letterbox e helpers do header
compartilhado, **sem editar `native/common.h`**.

## Validação: limites fixos

Mesmas oito entradas e matcher da rodada 2: score ≥ 0,25, mesma contagem e
classe, pareamento guloso pelo maior IoU, IoU ≥ 0,95, diferenças nos pares.
FP32: caixas ≤ 0,5 px, score ≤ 0,001 contra eager FP32 estrito. FP16: caixas
≤ 2 px, score ≤ 0,005 contra eager FP16. Nenhuma tolerância foi alterada.

| Variante | PASS / FAIL | Caixa máxima px | Score máximo | IoU mínimo | `detect --check bus` |
|---|---:|---:|---:|---:|---|
| NVIDIA FP32 | 8 / 0 | 0,000061 | 0,000001 | 0,999999 | PASS, IoU 0,999281 |
| NVIDIA FP16 | 8 / 0 | 0,500000 | 0,003906 | 0,996889 | PASS, IoU 0,997326 |
| Intel FP32 | 8 / 0 | 0,000122 | 0,000001 | 0,999999 | PASS, IoU 0,999281 |
| Intel FP16 | **6 / 2** | 0,500000 | **0,009277** | 0,994152 | PASS, IoU 0,994638 |
| CPU FP32 | 8 / 0 | 0,000061 | 0,000001 | 0,999999 | PASS, IoU 0,999281 |

Máximos arredondados como nas linhas C++; os dois caminhos Vulkan resident e
host têm os mesmos resultados. Ao todo: **68 PASS / 4 FAIL em 72 casos**, pois
os dois FAILs Intel se repetem em cada modo. Os cinco detects têm cinco
detecções cada e exigem IoU ≥ 0,90 em coordenadas originais, após letterbox C++.
Quatro entradas webcam não têm detecções nos dois lados: nesses casos os
máximos são zero e IoU 1 é uma convenção, não evidência geométrica.

FAILs literais Intel FP16 (resident; host idêntico):

```text
FAIL bus fp16 boxes_max=0.500000 scores_max=0.005127 dets=5/5 min_iou=0.996849 overall_max=468.437500 reason=score tolerance exceeded;
FAIL webcam_05 fp16 boxes_max=0.000000 scores_max=0.009277 dets=1/1 min_iou=1.000000 overall_max=633.132812 reason=score tolerance exceeded;
```

### Investigação do FP16 Intel

A contagem/classes e caixas passam; falha a aproximação dos scores. Mantendo
pesos e referências fixos, a alteração **somente de
`use_fp16_arithmetic=1` para 0** (`--storage-only`, storage/packing half mantidos)
passou nas oito entradas: caixa máxima 0,5 px, score máximo 0,001221, IoU mínimo
0,994152. Desligar somente Winograd deixou um FAIL (`webcam_05`, score 0,005371);
desligar somente SGEMM também deixou um FAIL (score 0,008789).

Isso associa a discrepância à política de aritmética half e seus kernels nesta
GPU, não a um erro de TopK, leitura ou tolerâncias. Não prova isoladamente um
bug no driver ou em um operador específico; trocar a opção também pode mudar
os caminhos de execução internos. A variante requerida **continua aritmética
FP16 ligada e continua FAIL**. O storage-only é diagnóstico, não uma décima
variante substituída na tabela de desempenho. Logs completos em
`artifacts/ncnn/{validate-*,detect-*,diagnostic-*}.txt`.

## Benchmark com controle térmico

`scripts/bench_ncnn.sh` roda **cinco variantes por padrão** em processos
separados, **três rodadas**, cooldown de **60 s antes de cada tentativa** e
check de host de cinco segundos via `uv run python cpu/bench.py --check-host`.
`--with-intel` acrescenta as quatro variantes Intel (nove no total), somente
por opt-in. Retenta somente exit 75 (concorrência), até cinco vezes além da
tentativa inicial, sempre com novo cooldown; erros de modelo/configuração
interrompem sem retentativa. O monitor original também acompanha o subprocesso
nativo: exclui somente o PID medido, equivalente à exclusão do próprio processo
em `cpu/bench.py`, sem mudar limiares, lista de compiladores ou excluir outros
trabalhos.

Ordem padrão, rodada 1: NVIDIA FP32 resident → NVIDIA FP16 resident → NVIDIA
FP32 host → NVIDIA FP16 host → CPU FP32. Rodadas 2 e 3 rotacionam por
deslocamentos **2 e 4**. Com `--with-intel`, as quatro Intel são acrescentadas
nessa lista e os deslocamentos são 4 e 8. O início remove resumo e blocos antigos
para impedir resumo obsoleto; não toca no diretório Intel preservado abaixo.
Só blocos aceitos são salvos em `artifacts/bench/round4-runs/`; todas as
tentativas ficam em `artifacts/bench/round4.txt`. Resumo em
`artifacts/bench/round4-summary.txt`. Medianas usam **três valores por variante**,
não um pool de 1500/600 iterações. Min/max são das três médias.

A sessão inicial de nove variantes foi interrompida pelo orquestrador por
**decisão do usuário de não continuar medindo a Intel**, não por erro do
benchmark. Os quatro blocos Intel da rodada 1 foram movidos para
`artifacts/bench/round4-intel-round1/`. Arquivos incompletos da rodada 2 e
log/resumo antigos foram removidos; a sessão final de cinco variantes começou
do zero. Não se misturaram resultados das sessões. Na execução final, o
orquestrador informou máquina ociosa/navegador fechado: **15 blocos aceitos,
zero descartes por concorrência**, sem conversão, build ou validação em paralelo.

### Cinco variantes controladas: medianas de três rodadas

| Variante | Mediana das médias ms | Mediana p95 ms | Mediana FPS | Min/max das médias ms |
|---|---:|---:|---:|---:|
| NVIDIA FP32 resident | 18,55 | 22,54 | 53,9 | 18,51–18,76 |
| NVIDIA FP16 resident | **9,69** | 11,66 | **103,2** | 9,47–9,83 |
| NVIDIA FP32 host | 19,19 | 22,63 | 52,1 | 18,98–19,24 |
| NVIDIA FP16 host | 10,17 | 12,84 | 98,4 | 10,05–10,19 |
| CPU FP32 host | 56,89 | 60,57 | 17,6 | 56,81–57,06 |

Métricas auxiliares, também medianas dos três blocos por variante. Medianas
dos componentes são tomadas separadamente, por isso não necessariamente somam
exatamente a mediana do total.

| Variante | Modelo ms | Pós-processamento ms | GPU processo MiB | RSS pico MiB | Carga ms | Primeiro ms |
|---|---:|---:|---:|---:|---:|---:|
| NVIDIA FP32 resident | 16,09 | 2,46 | 196,0 | 381,0 | 537,52 | 21,87 |
| NVIDIA FP16 resident | 7,42 | 2,26 | 128,0 | 362,7 | 582,63 | 12,01 |
| NVIDIA FP32 host | 18,28 | 0,90 | 196,0 | 386,0 | 535,98 | 29,92 |
| NVIDIA FP16 host | 9,32 | 0,85 | 128,0 | 368,7 | 574,98 | 19,89 |
| CPU FP32 host | 55,99 | 0,89 | n/a | 223,0 | 31,85 | 65,38 |

Em resident, o pós-processamento de 2,26–2,46 ms **inclui o download** da saída
bruta; não é só o custo dos algoritmos TopK. Em host, a transferência já está
no modelo e o pós-processamento custa 0,85–0,90 ms. O total host é 0,64 ms
maior em FP32 e 0,48 ms maior em FP16 que resident. O tamanho do par de modelo
é 27,769 MiB em todas as variantes. O RSS bem menor que eager/AOTI não é uma
medida isolada de pesos nem de pico de VRAM.

### Intel iGPU: uma única rodada, sem mediana

Os resultados de validação Intel acima permanecem válidos: FP32 **8/8 PASS**;
FP16 arithmetic **6/8 PASS**, com dois FAILs de score. Não foram descartados
por desempenho. Estes quatro benchmarks são **uma única rodada, sem mediana**,
com cooldown de 60 s, check/monitor de host e 50 aquecimentos + 500 medidas.
Não devem ser apresentados como três rodadas concluídas.

| Intel, rodada 1 | Média ms | p50 ms | p95 ms | it/s | Modelo ms | Pós-processamento ms |
|---|---:|---:|---:|---:|---:|---:|
| FP32 resident | 432,97 | 432,57 | 433,97 | 2,3 | 431,24 | 1,73 |
| FP16 resident (**FAIL estrito**) | 229,70 | 229,48 | 230,52 | 4,4 | 228,05 | 1,65 |
| FP32 host | 436,97 | 437,31 | 439,75 | 2,3 | 436,06 | 0,91 |
| FP16 host (**FAIL estrito**) | 230,50 | 230,30 | 231,34 | 4,3 | 229,64 | 0,85 |

**O benchmark Intel foi interrompido por decisão do usuário (Nicolas), porque
é lento demais para este modelo neste notebook; as rodadas controladas Intel
não foram concluídas. Isso é uma decisão de escopo, não uma falha de execução.**
Cada bloco Intel leva cerca de 2–4 minutos só para as inferências. Mesmo o
melhor caso, FP16 a 229,70 ms, é aproximadamente **15,4× mais lento que CPU
OpenVINO INT8 a 14,91 ms**; FP32 a 432,97 ms é 29,0× mais lento. Contra eager
CPU FP32 a 55,56 ms, são aproximadamente **4–8×**; contra OpenVINO CPU FP32
default a 39,84 ms, **5,8–10,9×**. Portanto a expressão 4–8× refere-se ao eager,
não a todos os runtimes CPU FP32. São comparações indicativas entre uma única
rodada Intel e medianas CPU de outra rodada, não estatística equivalente.

### Cooperative matrix NVIDIA: uso efetivo e A/B separado

Além de `cooperative_matrix_supported=1`, a linha `settings:` efetiva registra
`cooperative_matrix=1`. Para verificar execução, uma sonda temporária
`LD_PRELOAD` interceptou `ncnn::Pipeline::create` e
`ncnn::VkCompute::record_pipeline` em um **detect FP16 separado**. O shader de
tipo 21 corresponde a `convolution_gemm_cm` no enum instalado do ncnn:

```text
TRACE shader_type=21 created=4 dispatched=4 convolution_gemm_cm
```

A sonda também capturou o SPIR-V desse shader; `spirv-dis` confirmou
`OpCapability CooperativeMatrixKHR`, `SPV_KHR_cooperative_matrix` e instruções
**`OpCooperativeMatrixMulAddKHR`**. O detect completou com PASS. Com
`--no-coop`, `settings: cooperative_matrix=0` e não houve criação/dispatch de
`convolution_gemm_cm`; o detect também passou. Assim há evidência do caminho
cooperative matrix **realmente despachado**, não apenas do suporte anunciado.
Isso não significa que todas as convoluções ou o FP32 usem esse caminho.
Código da sonda, logs e SPIR-V estão em `artifacts/ncnn/trace-coop*` e
`artifacts/ncnn/convolution_gemm_cm.*`, fora do Git; a sonda **não foi usada
em nenhuma medição de latência**.

Depois da sessão controlada, em processos separados e sem a sonda, foram
executados um benchmark `--no-coop` e outro com a opção ligada, nessa ordem,
NVIDIA FP16 resident. **São execuções únicas não controladas, sem mediana e
sem cooldown entre os dois casos**, não parte da tabela controlada. Cada
processo manteve o protocolo 50+500; o check de host anterior passou.

| A/B NVIDIA FP16 resident, execução única não controlada | Média total ms | p50 ms | p95 ms | Modelo ms | Pós-processamento ms |
|---|---:|---:|---:|---:|---:|
| `--no-coop` (`cooperative_matrix=0`) | 13,00 | 12,85 | 14,49 | 10,72 | 2,27 |
| padrão (`cooperative_matrix=1`) | 9,77 | 9,27 | 13,30 | 7,53 | 2,24 |

O total caiu 24,8% (1,33×) e o componente modelo 29,8% nessa dupla. É coerente
com o uso observado, mas ordem, boost e térmica impedem tratar esse A/B
isolado como ganho causal estimado com o mesmo rigor das medianas controladas.

## Comparação com as rodadas anteriores

Números abaixo **citados, não reexecutados**, de
[`tensorrt-results.md`](tensorrt-results.md) e
[`cpu-results.md`](cpu-results.md). GPU anterior: entrada residente, 50+500,
sequência única; CPU anterior: mediana de três rodadas com cooldown, 20+200.
A rodada ncnn acrescenta download/top-k externos ao total, inclusive resident.

| Backend anterior | FP32 ms | FP16 ms | INT8 ms | Validação |
|---|---:|---:|---:|---|
| eager Python NVIDIA | 9,86 | 5,59 | — | referência |
| AOTInductor C++ NVIDIA | 8,54 | 3,89 | — | 8/8 em cada precisão |
| TensorRT C++ NVIDIA padrão | 7,44 | 2,95 | — | FP32: 7/8; FP16: 8/8 |
| TensorRT C++ NVIDIA sem TF32 | 7,87 | — | — | 8/8 |
| eager CPU | 55,56 | — | — | 8/8 |
| ONNX Runtime CPU | 43,63 | — | — | 8/8 |
| OpenVINO CPU explícito | 40,20 | — | 14,91 | 8/8; INT8 mixed: −0,3094 ponto mAP50–95 |
| OpenVINO CPU default | 39,84 | — | — | 8/8 |

### Conclusão

Na NVIDIA, ncnn FP16 resident é **1,91× mais rápido que ncnn FP32**
(18,55/9,69), mas seu total ainda é **1,73× mais lento que eager CUDA FP16**,
**2,49× que AOTI FP16** e **3,28× que TensorRT FP16** a 2,95 ms. FP32 resident
é **1,88× mais lento que eager GPU FP32** e **2,36× que TRT FP32 sem TF32**.
O TRT FP32 padrão tem um FAIL estrito e não deve ser usado como substituto
numericamente validado. O componente modelo ncnn FP16 isolado é 7,42 ms,
também acima dos 2,95 ms do TRT que já incluem seu TopK interno: o download e
pós-processamento externo explicam parte, mas não toda a diferença.

O benefício demonstrado de **Vulkan independente de fabricante** é executar
os mesmos pesos e o mesmo binário C++ nas duas GPUs, sem CUDA, libtorch ou
plugins, com FP32 estrito aprovado em ambas. Isso **não trouxe vantagem de
latência sobre CUDA/TensorRT na NVIDIA** neste modelo. Cooperative matrix
é realmente usada em parte da rede FP16 e ajuda no A/B exploratório, mas não
elimina essa distância. A carga ncnn inclui criação de pipelines e ficou em
aproximadamente 0,54–0,58 s na NVIDIA; não se deve compará-la isoladamente com
construção offline de engines TensorRT ou compilação AOTI.

**ncnn CPU FP32 a 56,89 ms não supera os runtimes CPU anteriores**: fica
próximo, mas 2,4% acima do eager CPU a 55,56 ms; é 30,4% mais lento que ORT e
42,8% mais lento que OpenVINO default a 39,84 ms. Contra OpenVINO INT8 a
14,91 ms, a latência é 3,82× maior, lembrando que INT8 é precisão distinta e
teve avaliação mAP própria na rodada 3. Retirar Python não basta para superar
as otimizações do OpenVINO.

**A iGPU não compensa para esta YOLOv10s neste notebook**: aproximadamente
2,3 it/s em FP32 e 4,4 it/s em FP16, com o FP16 ainda FAIL nos limites de
score. A CPU já entrega 17,6 it/s com ncnn FP32, 25,1 com OV FP32 default e
67,1 com OV INT8. A decisão de encerrar as medições Intel evita investir
mais tempo em um caminho sem vantagem prática observada, sem transformar
uma única rodada em uma mediana nem esconder os resultados de validação.

## Reprodução e verificações

Preparar os artefatos conforme o design, da origem somente leitura. Não rodar
`aoti/export.py` nem regenerar `artifacts/ref`.

```bash
uv sync
uv run python ncnn/export_ncnn.py
cmake --preset release
cmake --build --preset release
build/bin/yolo_ncnn check-postprocess artifacts/ncnn
for device in nvidia intel cpu; do
  for precision in fp32 fp16; do
    [ "$device/$precision" = cpu/fp16 ] && continue
    for mode in resident host; do
      [ "$device/$mode" = cpu/resident ] && continue
      taskset -c 0,2,4,6,8,10 build/bin/yolo_ncnn validate \
        artifacts/ncnn/yolov10s.ncnn.param artifacts/ncnn/yolov10s.ncnn.bin \
        artifacts/ref --device "$device" --precision "$precision" --mode "$mode"
      echo "validate $device $precision $mode exit=$?" # Intel FP16 retorna 1
    done
    suffix=eager_dets
    [ "$precision" = fp16 ] && suffix=eager_fp16_dets
    build/bin/yolo_ncnn detect artifacts/ncnn/yolov10s.ncnn.param \
      artifacts/ncnn/yolov10s.ncnn.bin artifacts/data/bus.jpg \
      --device "$device" --precision "$precision" \
      --check "artifacts/ref/bus.$suffix.txt"
  done
done
# Executar somente após exportação, build e validações, sem trabalho pesado paralelo.
bash scripts/bench_ncnn.sh  # padrão: NVIDIA + CPU, cinco variantes
# Opt-in para uma nova sessão completa com Intel (não executada após a decisão):
# bash scripts/bench_ncnn.sh --with-intel
ldd build/bin/yolo_ncnn
uv run python -m py_compile ncnn/export_ncnn.py
bash -n scripts/bench_ncnn.sh
git diff --check
```

Também foram executados os três diagnósticos Intel FP16 descritos acima;
teste sintético do resumo (mediana, min/max das três médias, nove variantes,
rejeição de rodada ausente), compilação dos dois blocos Python embutidos no
shell e teste do wrapper que aceita um filho bem-sucedido/preserva exit 23 de
um erro do modelo. Após tornar Intel opt-in, testes conferiram cinco/nove
variantes únicas, ordens rotativas completas/distintas e rejeição de flag
inválida antes de limpar os artefatos. O build emite somente avisos já
conhecidos do stb (inicializadores/funções estáticas não usadas), sem suprimir
checks.

## Limitações

Oito entradas, quatro vazias, não substituem COCO mAP nem garantem equivalência
em toda a distribuição. Não se atribui ao Intel FP16 FAIL a condição de
substituto numericamente validado, mesmo com detect bus aprovado. Não foram
medidos INT8/BF16 ncnn, storage-only no benchmark nem webcam em tempo real.

A iGPU **compartilha memória, potência e orçamento térmico com a CPU**. O
notebook mantém boost/clocks dinâmicos: não se travaram clocks, potência ou
ventoinhas, nem se confirmou alimentação pela tomada por software. Cooldown
reduz, mas não garante temperaturas iniciais idênticas. Ordem rotativa é
determinística, não aleatória; três rodadas e min/max não são intervalos de
confiança. O monitor detecta compiladores e carga externa de CPU, mas não é
um detector completo de trabalhos exclusivamente GPU. O desktop pode estar
ativo. Caches do driver/sistema não foram apagados. As tabelas anteriores têm
protocolos e datas de execução distintos; comparações são deste notebook e
não um ranking universal entre APIs ou fabricantes.
