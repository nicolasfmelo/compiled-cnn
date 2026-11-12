# PBI-095 — YOLOv10s em CPU, rodada 3

Execução: 11/11/2026. Design aprovado em
[`2026-11-11-cpu-baseline-design.md`](superpowers/specs/2026-11-11-cpu-baseline-design.md).
Mesmo checkpoint e ONNX FP32 da rodada 2, modelo fundido end-to-end, batch 1,
entrada `1×3×640×640`, saída `1×300×6`, sem NMS externo. As referências eager
FP32 geradas na GPU foram **copiadas, nunca regeneradas**.

**Resultado principal:** OpenVINO FP32 explícito mediu **43,43 ms / 23,0 it/s**;
OpenVINO INT8, **19,35 ms / 51,7 it/s** (2,24× o OV FP32). As cinco variantes
passaram nas oito entradas. O INT8 final usa NNCF `mixed`, sem `ignored_scope`,
e perdeu **0,3094 ponto de mAP50–95**, abaixo do limite de 1,0 ponto. Não houve
alteração de tolerâncias. Duas tentativas anteriores de quantização falharam
no critério de detecção e estão registradas abaixo.

## Ambiente e dependências

| Componente | Versão / configuração |
|---|---|
| CPU | Intel Core i5-13450HX, 6 P-cores com SMT + 4 E-cores, 16 CPUs lógicas |
| Topologia verificada em sysfs | pares SMT `0–1`, `2–3`, `4–5`, `6–7`, `8–9`, `10–11`; E-cores `12–15` |
| ISA | AVX2, FMA, F16C e AVX-VNNI; sem AVX-512 ou AMX |
| Cache | L2 9,5 MiB, L3 20 MiB (`lscpu`) |
| RAM / swap | 15 GiB / 15 GiB; swap já ocupada por outros trabalhos, sem limpeza forçada |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128, inferência nesta rodada em CPU FP32 |
| Ultralytics / ultralytics-thop | 8.4.210 / 2.2.5 |
| ONNX Runtime | 1.31.1, `CPUExecutionProvider` |
| OpenVINO | 2026.4.1 (`2026.4.1-22982-e213a147257-releases/2026/4`) |
| NNCF | 3.4.0 |
| pycocotools | 2.0.11 |
| NumPy / NetworkX | 2.4.6 / 3.6.1 |

Foram adicionadas somente três dependências diretas: `openvino` (conversão,
IR e execução em CPU), `nncf` (quantização pós-treino) e `pycocotools`
(avaliação COCO padrão). O resolvedor precisou trocar NumPy 2.5.4 por 2.4.6
e NetworkX 3.7 por 3.6.1: **NNCF 3.4.0 exige `numpy<2.5.0` e
`networkx<=3.6.1`**. Não foi um downgrade arbitrário para contornar testes.
As demais adições do lockfile são transitivas dessas três bibliotecas;
Torch, ORT e Ultralytics mantiveram as versões. `psutil`, já presente como
transitiva antes desta rodada, é usado para detectar concorrência no benchmark.

## Implementação e protocolo

- `cpu/common.py`: runtimes, leitura dos binários fixos, preprocessamento
  compartilhado com `aoti.export`, matcher e impressão de validação.
- `cpu/validate.py`: cinco subprocessos, oito amostras por variante; retorna 1
  se qualquer critério falhar, sem interromper as variantes seguintes.
- `cpu/quantize_int8.py`: conversão `ov.convert_model` do ONNX fixo, gravação
  dos dois IRs e calibração NNCF; opções para reproduzir as tentativas.
- `cpu/eval_map.py`: três subprocessos e avaliação restrita aos mesmos 500 IDs.
- `cpu/bench.py` e `native/bench_cpu.sh`: cinco processos sequenciais, saída
  em stdout e `artifacts/bench/round3.txt`; metadados e medidas individuais
  em `artifacts/cpu/`, fora do Git.

O modelo eager é carregado pelo helper existente e imediatamente convertido
com `.cpu().float()`; não se alterou `aoti/`. O helper move transitoriamente o
modelo para CUDA durante a carga: esse custo está no tempo de carga eager,
**não há inferência ou benchmark de GPU nesta rodada**. Imports ficam fora do
tempo de carga, mas dentro do RSS do processo. Para OV, carga inclui leitura
do IR, compilação e criação do request; para ORT, criação da sessão e suas
otimizações. Não se usou cache persistente de compilação OV.

### Threads e afinidade

Eager: seis threads intra-op, uma inter-op, `inference_mode`. ORT: seis
intra-op, uma inter-op, `ORT_SEQUENTIAL`, `ORT_ENABLE_ALL` e somente CPU EP.
Ambos e os dois OV explícitos têm afinidade externa `0,2,4,6,8,10`.

OV explícito: `INFERENCE_PRECISION_HINT=f32`, `PERFORMANCE_HINT=LATENCY`,
`INFERENCE_NUM_THREADS=6`, `ENABLE_HYPER_THREADING=False`,
`ENABLE_CPU_PINNING=True`. O IR INT8 mantém suas operações quantizadas mesmo
com a dica f32. Uma inspeção do grafo compilado confirmou precisões de runtime
`u8`/`i8` além de `f32`/`i32`; não se está chamando um modelo inteiramente FP32
de INT8.

OV default recebe **nenhuma propriedade** e **nenhum taskset**. As propriedades
lidas de volta foram, nesta máquina/versão, as mesmas do OV explícito:
f32, LATENCY, seis threads, HT desligado, pinning ligado, um stream. Isso não
significa afinidade externa igual: o processo default tem CPUs `0–15`
disponíveis, e a seleção interna de workers/SMT pode diferir.

### Inferência, frequência e pico

Entrada de bus preparada uma vez, inclusive o wrapper de tensor eager antes
das iterações medidas. Primeiro forward separado, **20 aquecimentos + 200
medidas**, chamadas síncronas, percentis com interpolação linear, it/s =
`1000/média`. Não entram leitura de imagem, letterbox, desenho, COCO ou cópia
para GPU. RSS é o pico de todo o processo, não só pesos. O tamanho em disco
é `.pt` para eager, ONNX para ORT e XML+BIN para OV; o checkpoint e os formatos
compilados não representam a mesma organização/precisão dos pesos em disco.

Ultralytics `get_flops(model_fundido, imgsz=640)` retornou **21,7026048 GFLOPs**,
MAC contado como duas operações. Uma chamada direta de THOP com tensor
`1×3×640×640` retornou **exatamente o mesmo número**, verificando a extrapolação
do profiler nessa versão. É o modelo fundido one-to-one realmente executado,
não o checkpoint com duas cabeças nem uma estimativa para resolução menor.

Uma thread amostra `scaling_cur_freq` a cada 100 ms durante as 200 medidas.
São mantidas as leituras exigidas das CPUs `0,2,4,6,8,10`. **Correção de
instrumentação encontrada durante o relatório:** no OV default, as leituras
desses irmãos SMT ociosos ficam baixas/defasadas, enquanto os irmãos ímpares
estão ativos. Um diagnóstico encontrou médias de 2,33 GHz nos pares e
3,91 GHz nos ímpares. Usar apenas os pares produzia um pico falsamente baixo
e mais de 100% para FP32. Por isso também se lê cada irmão ímpar, tomando o
máximo de cada par e depois a média dos seis núcleos. A média somente dos
pares continua impressa; todas as variantes usam a mesma instrumentação.
Trata-se de uma estimativa por sysfs, não de contadores APERF/MPERF nem clocks
travados. A sequência completa foi repetida após essa correção.

`pico_FP32 = 6 × 32 FLOP/ciclo × GHz`; `GFLOPS_equivalentes = 21,7026048 / latência_s`.
Para INT8, **o percentual é equivalente contra o pico FP32**, não utilização
física de unidades FP32 ou percentual do pico INT8. Pode legitimamente
ultrapassar 100% com AVX-VNNI. Não se estimou o teto INT8.

## Quantização: calibração e tentativas

COCO val2017 ordenado por ID: calibração usa **300 imagens, posições 501–800**,
IDs 57027–93353. Avaliação usa posições 1–500, IDs 139–56545, sem interseção.
Preprocessamento idêntico ao eager/exportador: letterbox 640×640, padding 114,
RGB, divisão por 255, NCHW FP32. `nncf.Dataset` lê as imagens sob demanda.
Não foi usado train2017 nem nenhuma das imagens de avaliação na calibração.

`target_device=CPU`, `subset_size=300`, correção rápida de bias padrão;
`model_type=None` — não se classificou a CNN como Transformer. Os IRs foram
salvos com **`compress_to_fp16=False`**, sem compressão silenciosa dos pesos
FP32. INT8 é uma rede mista: algumas operações continuam FP32.

| Tentativa | Preset | Ignored scope | Tempo NNCF (s) | Detecção | mAP50–95 | mAP50 | Queda vs OV FP32 (pontos) |
|---|---|---|---:|---|---:|---:|---:|
| Inicial | performance | nenhum | 108,986 | 5 PASS / 3 FAIL | 48,9240 | 65,0555 | 0,4032 — PASS mAP |
| Pós-processamento excluído | performance | 23 nós TopK e descendentes | 61,424 | 5 PASS / 3 FAIL | 43,6109 | 64,2547 | 5,7163 — FAIL mAP |
| Mixed, primeira execução | mixed | nenhum | 41,579 | 8 PASS / 0 FAIL | 49,0178 | 65,3091 | 0,3094 — PASS |
| **Final, comando sem flags** | **mixed** | **nenhum** | **38,605** | **8 PASS / 0 FAIL** | **49,0178** | **65,3091** | **0,3094 — PASS** |

Tempos medem somente `nncf.quantize`, sem download, conversão ONNX→IR ou
serialização. Execuções posteriores têm caches de arquivos/bibliotecas mais
quentes; não interpretar a diferença de tempo como aceleração causal do preset.

A tentativa de `IgnoredScope` encontrou os dois TopK no grafo convertido e
excluiu seus descendentes por nome:

```text
node_And_173, node_Cast_172, node_Cast_174, node_Div_175,
node_Equal_170, node_Less_167, node__to_copy, node_expand,
node_expand_1, node_floor_divide, node_gather, node_gather_1,
node_gather_2, node_remainder, node_topk_1__1, node_topk__1,
node_unsqueeze_1, node_unsqueeze_2, node_unsqueeze_3,
node_unsqueeze_4, node_view_10, output0, output0/sink_port_0
```

NNCF confirmou os 23 nomes. Excluir seleção/empacotamento **não corrigiu os
erros de score** e piorou caixas e mAP. Portanto a configuração final **não
ignora nenhum nó**. Não se afirma ter isolado um único operador responsável
pela piora: o escopo também altera onde a quantização é aplicada no grafo.
O preset `mixed` (pesos simétricos, ativações assimétricas) resolveu os três
erros de score sem essa exclusão e foi escolhido pelos dois critérios de
acurácia, não por um benchmark mais favorável.

| IR final | Bytes XML+BIN | MiB |
|---|---:|---:|
| FP32, `artifacts/yolov10s_fp32_ov/` | 29.381.871 | 28,021 |
| INT8, `artifacts/yolov10s_int8_ov/` | 8.097.298 | 7,722 |

Redução de tamanho: 72,4%. A tentativa performance sem escopo tinha 8.097.492
bytes; com escopo, 8.103.453 bytes. Logs/JSON das tentativas foram preservados
em `artifacts/`, não versionados.

## Validação nas oito entradas

FP32 porta exatamente `native/common.h`: filtra score ≥ 0,25, exige contagens
iguais (retorno antecipado em contagem diferente), toma gulosa e globalmente
o maior IoU restante **independentemente da classe** e só depois verifica
classe e IoU ≥ 0,95. Diferenças de caixas/scores são calculadas nos pares,
com limites inclusivos **0,5 px / 0,001**. `overall_max` continua linha a linha
e apenas informativo; TopK pode permutar linhas de baixa confiança. Em dois
conjuntos vazios, máximos são zero e IoU é convencionalmente 1.

INT8 usa o pareamento geométrico, mesma classe, IoU ≥ 0,80 e diferença de
score ≤ 0,05. Detecções restantes dos dois lados são impressas; só passam se
score < 0,30. Não se aplica o limite FP32 de 0,5 px ao INT8. Não houve nenhum
unmatched na configuração final. Na tentativa performance, `webcam_00` teve
um unmatched real de classe 59, score 0,261125: aceito pelo critério explícito,
sem esconder a diferença de contagem.

| Variante final | PASS / FAIL | Caixa máxima nos pares (px) | Score máximo nos pares | IoU mínimo |
|---|---|---:|---:|---:|
| eager-cpu FP32 | 8 / 0 | 0,000030518 | 0,000000596 | 0,999999404 |
| ort-cpu FP32 | 8 / 0 | 0,000061035 | 0,000000656 | 0,999999404 |
| ov-cpu FP32 | 8 / 0 | 0,000061035 | 0,000000775 | 0,999999285 |
| ov-cpu INT8 mixed | 8 / 0 | 2,957703 | 0,034189433 | 0,958209634 |
| ov-cpu FP32 default | 8 / 0 | 0,000061035 | 0,000000775 | 0,999999285 |

A sanidade eager CPU contra referência GPU passou estritamente; não foi
necessário investigar discrepâncias CPU/GPU além das permutações informativas.
Saída literal final do INT8:

```text
PASS bus ov-cpu-int8 boxes_max=2.957703 scores_max=0.015361 dets=5/5 min_iou=0.958210 overall_max=489.579254
PASS zidane ov-cpu-int8 boxes_max=1.682800 scores_max=0.015832 dets=3/3 min_iou=0.967715 overall_max=596.140198
PASS webcam_00 ov-cpu-int8 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=576.704163
PASS webcam_01 ov-cpu-int8 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=632.180969
PASS webcam_02 ov-cpu-int8 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=572.192017
PASS webcam_03 ov-cpu-int8 boxes_max=0.000000 scores_max=0.000000 dets=0/0 min_iou=1.000000 overall_max=633.075439
PASS webcam_04 ov-cpu-int8 boxes_max=1.425903 scores_max=0.034189 dets=1/1 min_iou=0.960818 overall_max=626.631958
PASS webcam_05 ov-cpu-int8 boxes_max=0.819916 scores_max=0.029577 dets=1/1 min_iou=0.974743 overall_max=633.483826
```

FAILs da primeira tentativa, mantidos honestamente:

```text
FAIL zidane ov-cpu-int8 boxes_max=1.763153 scores_max=0.058680 dets=3/3 min_iou=0.979004 overall_max=628.625977 reason=score tolerance exceeded;
FAIL webcam_04 ov-cpu-int8 boxes_max=4.369629 scores_max=0.080760 dets=1/1 min_iou=0.926733 overall_max=626.006592 reason=score tolerance exceeded;
FAIL webcam_05 ov-cpu-int8 boxes_max=0.702942 scores_max=0.053067 dets=1/1 min_iou=0.981735 overall_max=636.175049 reason=score tolerance exceeded;
```

Com escopo ignorado, os mesmos três scores falharam; os IoUs caíram para
0,867016 / 0,863304 / 0,845524, respectivamente. Nenhuma tolerância foi relaxada.
`cpu/test_common.py` passou em cinco testes, incluindo permutação, erros de
caixa/score/classe/contagem, conjuntos vazios, fronteira unmatched 0,30 e
inversão de letterbox com filtro 0,001.

## COCO mAP: mesmas 500 imagens

Todas as até 300 saídas com score ≥ 0,001 são mantidas, sem NMS. Desfaz-se
letterbox com o mesmo ganho e padding inteiro do preprocessamento, limita-se
a caixa às dimensões originais, converte-se xyxy→xywh e mapeia-se classe
0–79 para os 80 IDs COCO, que têm lacunas. `COCOeval` recebe somente os 500
`imgIds` escolhidos. Usa-se a métrica COCO bbox padrão, com `maxDets=[1,10,100]`;
isto não significa pré-cortar as 300 predições para 100 antes do avaliador.

| Variante | mAP50–95 (%) | mAP50 (%) | Queda mAP50–95 vs OV FP32 |
|---|---:|---:|---:|
| eager-cpu FP32 | 49,3272 | 65,2649 | — |
| ov-cpu FP32 | 49,3272 | 65,2649 | referência |
| ov-cpu INT8 mixed | 49,0178 | 65,3091 | **0,3094 ponto — PASS (≤ 1,0)** |

Eager e OV FP32 concordam até muito além das casas mostradas. Os ~46,3% oficiais
são no val2017 completo; 49,33% neste subconjunto determinístico não é a mesma
população e não é evidência de ganho sobre o modelo oficial. A pequena subida
de mAP50 no INT8 também não demonstra melhora geral. O primeiro desenvolvimento
do script encontrou um erro de serialização de IDs NumPy após avaliar eager;
foi corrigido convertendo os IDs para `int`, e a avaliação completa foi repetida.

## Benchmark final das cinco variantes

Uma única sequência final, na ordem abaixo, processo independente por variante.
Antes dela: `loadavg: 2.02 4.92 8.58`; os valores de 5/15 minutos ainda refletiam
trabalho já encerrado. O check amostrou cinco segundos e não encontrou builds
nem CPU externa agregada >75% de um núcleo lógico. Durante cada processo, um
monitor de 500 ms continuou verificando compiladores e CPU externa. A sequência
final não disparou o monitor. Desktop permaneceu ativo; não se promete uma
máquina absolutamente isolada.

Houve uma primeira sequência com Rust/Clippy concorrente, descartada, e várias
tentativas recusadas antes de encontrar uma janela livre. Uma segunda sequência
completa foi substituída após o diagnóstico dos irmãos SMT da frequência.
**A tabela abaixo não mistura variantes dessas sequências.** Não houve download,
calibração ou mAP durante o benchmark final.

| Variante | Média ms | p50 ms | p95 ms | it/s | RSS pico MiB | Carga ms | Primeiro ms | Modelo MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eager-cpu FP32 | 58,29 | 57,02 | 68,46 | 17,2 | 933,9 | 241,19 | 63,98 | 15,853 |
| ort-cpu FP32 | 46,43 | 45,37 | 53,44 | 21,5 | 217,2 | 76,27 | 43,75 | 28,358 |
| ov-cpu FP32 | **43,43** | 42,61 | 48,88 | 23,0 | 214,4 | 126,65 | 47,94 | 28,021 |
| ov-cpu INT8 | **19,35** | 18,98 | 21,60 | 51,7 | 169,6 | 269,21 | 17,48 | 7,722 |
| ov-cpu FP32 default | 55,19 | 54,10 | 61,16 | 18,1 | 213,5 | 133,85 | 40,31 | 28,021 |

| Variante | GHz, média dos 6 P-cores (máximo por par SMT) | GHz, somente CPUs pares | Pico FP32 GFLOPS | GFLOPS equivalentes atingidos | % do pico FP32 |
|---|---:|---:|---:|---:|---:|
| eager-cpu FP32 | 4,210 | 4,175 | 808,36 | 372,32 | 46,06% |
| ort-cpu FP32 | 3,897 | 3,878 | 748,28 | 467,43 | 62,47% |
| ov-cpu FP32 | 3,712 | 3,703 | 712,67 | 499,77 | 70,13% |
| ov-cpu INT8 | 3,361 | 3,358 | 645,38 | 1121,54 | **173,78% equivalente, não utilização FP32** |
| ov-cpu FP32 default | 3,086 | 1,297 | 592,49 | 393,20 | 66,36% |

Saída literal dos cinco blocos de `bash native/bench_cpu.sh`:

```text
variant: eager-cpu-fp32
settings: threads=6 interop_threads=1 inference_mode=on affinity=0,2,4,6,8,10 warmup=20 iterations=200
latency ms: mean 58.29 | p50 57.02 | p95 68.46
throughput: 17.2 it/s
peak RSS MiB: 933.9
load ms: 241.19 | first run ms: 63.98
model MiB: 15.853
frequency: P-core mean 4.210 GHz (max per SMT pair) | even CPUs mean 4.175 GHz | samples 116
compute: model 21.703 GFLOPs | achieved 372.32 GFLOPS | peak FP32 808.36 GFLOPS | 46.06% of peak
variant: ort-cpu-fp32
settings: provider=CPUExecutionProvider intra_threads=6 inter_threads=1 ORT_SEQUENTIAL ORT_ENABLE_ALL affinity=0,2,4,6,8,10 warmup=20 iterations=200
latency ms: mean 46.43 | p50 45.37 | p95 53.44
throughput: 21.5 it/s
peak RSS MiB: 217.2
load ms: 76.27 | first run ms: 43.75
model MiB: 28.358
frequency: P-core mean 3.897 GHz (max per SMT pair) | even CPUs mean 3.878 GHz | samples 92
compute: model 21.703 GFLOPs | achieved 467.43 GFLOPS | peak FP32 748.28 GFLOPS | 62.47% of peak
variant: ov-cpu-fp32
settings: INFERENCE_PRECISION_HINT=<Type: 'float32'> PERFORMANCE_HINT=LATENCY INFERENCE_NUM_THREADS=6 ENABLE_HYPER_THREADING=False ENABLE_CPU_PINNING=True NUM_STREAMS=1 affinity=0,2,4,6,8,10 warmup=20 iterations=200
latency ms: mean 43.43 | p50 42.61 | p95 48.88
throughput: 23.0 it/s
peak RSS MiB: 214.4
load ms: 126.65 | first run ms: 47.94
model MiB: 28.021
frequency: P-core mean 3.712 GHz (max per SMT pair) | even CPUs mean 3.703 GHz | samples 87
compute: model 21.703 GFLOPs | achieved 499.77 GFLOPS | peak FP32 712.67 GFLOPS | 70.13% of peak
variant: ov-cpu-int8
settings: INFERENCE_PRECISION_HINT=<Type: 'float32'> PERFORMANCE_HINT=LATENCY INFERENCE_NUM_THREADS=6 ENABLE_HYPER_THREADING=False ENABLE_CPU_PINNING=True NUM_STREAMS=1 affinity=0,2,4,6,8,10 warmup=20 iterations=200
latency ms: mean 19.35 | p50 18.98 | p95 21.60
throughput: 51.7 it/s
peak RSS MiB: 169.6
load ms: 269.21 | first run ms: 17.48
model MiB: 7.722
frequency: P-core mean 3.361 GHz (max per SMT pair) | even CPUs mean 3.358 GHz | samples 39
compute: model 21.703 GFLOPs | achieved 1121.54 GFLOPS | peak FP32 645.38 GFLOPS | 173.78% of peak
variant: ov-cpu-fp32-default
settings: INFERENCE_PRECISION_HINT=<Type: 'float32'> PERFORMANCE_HINT=LATENCY INFERENCE_NUM_THREADS=6 ENABLE_HYPER_THREADING=False ENABLE_CPU_PINNING=True NUM_STREAMS=1 affinity=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 warmup=20 iterations=200
latency ms: mean 55.19 | p50 54.10 | p95 61.16
throughput: 18.1 it/s
peak RSS MiB: 213.5
load ms: 133.85 | first run ms: 40.31
model MiB: 28.021
frequency: P-core mean 3.086 GHz (max per SMT pair) | even CPUs mean 1.297 GHz | samples 110
compute: model 21.703 GFLOPs | achieved 393.20 GFLOPS | peak FP32 592.49 GFLOPS | 66.36% of peak
```

## Comparação com as rodadas GPU e conclusão

Números **citados, não reexecutados**, de
[`tensorrt-results.md`](tensorrt-results.md), RTX 3050 Laptop:

| Runtime GPU | FP32 ms | FP16 ms |
|---|---:|---:|
| eager Python | 9,86 | 5,59 |
| AOTInductor C++ | 8,54 | 3,89 |
| TensorRT C++ padrão | 7,44 (**1 FAIL estrito**) | 2,95 |
| TensorRT C++ sem TF32 | 7,87 (8/8 PASS) | — |

A GPU mantém vantagem clara: OV CPU FP32 é 4,40× mais lento que eager GPU
FP32 e 5,52× mais lento que TRT FP32 sem TF32. Mesmo INT8 CPU é 6,56× mais
lento que TRT GPU FP16. São políticas de precisão e protocolos diferentes
(GPU: 50+500, CPU: 20+200), não um teste controlado de hardware igual.

Para um futuro engine C++ próprio, os alvos medidos são **43,43 ms em FP32** e
**19,35 ms em INT8**, preservando os critérios de detecção e mAP. OV FP32 é
1,34× mais rápido que eager CPU; INT8 reduz mais 55,4% da latência sobre OV
FP32. O default sem taskset foi 27,1% mais lento que o explícito, embora as
propriedades efetivas fossem iguais; a frequência e a escolha de irmãos SMT
não foram iguais. Não se atribui toda a diferença exclusivamente aos E-cores,
que o default não precisou usar para seis threads com HT desligado.

A execução FP32 mais rápida atingiu **499,77 de 712,67 GFLOPS estimados (70,13%)**.
Um limite puramente aritmético àquela frequência seria aproximadamente
**30,45 ms**, deixando um fator máximo matemático de 1,43× / 29,9% de redução
até 100%; não é promessa atingível. Memória, ativações, atenção, TopK, conversões
e overhead de escalonamento não consistem só em FMAs densas. O INT8 já excede
o teto FP32 em operações equivalentes: **não se deduz sua distância ao teto
INT8** a partir desse percentual.

Limitações: notebook híbrido com boost e térmica variáveis, sem fixar clocks,
potência, governador ou confirmar alimentação pela tomada; ordem não
randomizada, uma sequência final e nenhum intervalo de confiança. As médias
de frequência mudam entre variantes e não são frequência nominal constante.
O amostrador de frequência e o monitor de concorrência são threads de baixo
custo, presentes em todos os processos. `scaling_cur_freq` é uma estimativa,
e o máximo por par SMT evita o caso ocioso observado, mas não substitui
contadores de ciclos. O pico FMA é um teto ideal, não um modelo de memória.
500 imagens não substituem o val2017 completo e oito entradas não garantem
equivalência em toda a distribuição. AOTI CPU, C++ CPU, FP16 CPU e GPU INT8
não foram avaliados nesta rodada.

## Reprodução exata

Na raiz da worktree, origem de artefatos **somente leitura**. Não executar
`aoti/export.py` nem regenerar `artifacts/ref`.

```bash
mkdir -p artifacts
cp -r /home/nicolas/Projects/experimentos/compiled-cnn/artifacts/{data,ref,yolov10s_fp32.onnx} artifacts/
cp /home/nicolas/Projects/experimentos/compiled-cnn/yolov10s.pt .
uv sync

# Cache reutilizável fora do repositório; não versionar datasets.
mkdir -p "$HOME/.cache/compiled-cnn/coco"
(
  cd "$HOME/.cache/compiled-cnn/coco"
  for item in zips/val2017.zip annotations/annotations_trainval2017.zip; do
    file=${item##*/}
    if [ ! -f "$file" ]; then
      curl --fail --location --retry 3 "http://images.cocodataset.org/$item" -o "$file" || \
        curl --fail --location --retry 3 "https://images.cocodataset.org/$item" -o "$file"
    fi
    unzip -n -q "$file"
  done
)

# Configuração FINAL selecionada: mixed, sem ignored_scope.
uv run python cpu/quantize_int8.py
uv run python cpu/validate.py
uv run python cpu/eval_map.py
# Aguardar outros builds, downloads e avaliações terminarem antes deste comando.
bash native/bench_cpu.sh

uv run python cpu/test_common.py
uv run python -m py_compile cpu/*.py
bash -n native/bench_cpu.sh
git diff --check
```

Para reproduzir as tentativas anteriores, **antes de reconstruir o INT8 final**:

```bash
uv run python cpu/quantize_int8.py --preset performance
uv run python cpu/validate.py  # retorna 1: três FAILs INT8, sem alterar tolerâncias
uv run python cpu/eval_map.py # retorna 0: perda mAP de 0,4032 ponto
uv run python cpu/quantize_int8.py --preset performance --ignore-postprocess
uv run python cpu/validate.py  # retorna 1: três FAILs INT8
uv run python cpu/eval_map.py # retorna 1: perda mAP de 5,7163 pontos
uv run python cpu/quantize_int8.py  # restaura mixed, sem ignored_scope
uv run python cpu/validate.py
uv run python cpu/eval_map.py
```

Verificação final executada: `uv sync` OK; quantização final OK; validação
**40 PASS / 0 FAIL**, oito linhas de resultado por variante; mAP completo
OK com queda de 0,3094 ponto; benchmark OK com cinco blocos na mesma sessão;
cinco testes sintéticos OK; compilação sintática Python, sintaxe Bash e
`git diff --check` OK. As recusas de benchmark por concorrência foram mantidas
como recusas, nunca convertidas em PASS; o primeiro erro de serialização no
avaliador foi corrigido e a verificação repetida.
