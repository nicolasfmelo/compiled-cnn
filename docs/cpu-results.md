# PBI-095 — YOLOv10s em CPU, rodada 3

Execução: 11/11/2026. Design aprovado em
[`2026-11-11-cpu-baseline-design.md`](superpowers/specs/2026-11-11-cpu-baseline-design.md).
Mesmo checkpoint e ONNX FP32 da rodada 2, modelo fundido end-to-end, batch 1,
entrada `1×3×640×640`, saída `1×300×6`, sem NMS externo. As referências eager
FP32 geradas na GPU foram **copiadas, nunca regeneradas**.

**Resultado de acurácia:** as cinco variantes passaram nas oito entradas.
O INT8 final usa NNCF `mixed`, sem `ignored_scope`, e perdeu **0,3094 ponto de
mAP50–95**, abaixo do limite de 1,0 ponto. Não houve alteração de tolerâncias.
Duas tentativas anteriores de quantização falharam no critério de detecção e
estão registradas abaixo. **O benchmark final usa medianas de três rodadas com
controle térmico, executadas pelo orquestrador:** OV FP32 default **39,84 ms**,
OV FP32 explícito **40,20 ms** e INT8 **14,91 ms**, ganho de **2,70× sobre OV FP32**.
O default não é mais lento que a configuração manual: a conclusão anterior
de penalidade de 27% era um artefato térmico de executar essa variante por último.

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
- `cpu/bench.py` e `scripts/bench_cpu.sh`: três rodadas de cinco processos, com
  cooldown e ordem rotativa; todos os blocos em stdout e `artifacts/bench/round3.txt`.
  Os 15 JSONs aceitos ficam em `artifacts/bench/round3-runs/`, e as medianas em
  `artifacts/bench/round3-summary.{txt,json}`, fora do Git.

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

INT8 usa o pareamento geométrico somente enquanto o maior IoU restante for
≥ 0,80; cada par ainda exige mesma classe e diferença de score ≤ 0,05.
Quando o melhor IoU fica abaixo de 0,80, o pareamento termina: todas as
restantes dos dois lados são impressas como unmatched e só passam se score
< 0,30. A revisão corrigiu um bug que forçava pares com qualquer sobreposição
positiva, rejeitando incorretamente duas detecções de score 0,26 com IoU 0,05.
O ramo FP32 não foi alterado: 250 comparações sintéticas determinísticas
produziram resultados exatamente iguais aos da implementação anterior.
Não se aplica o limite FP32 de 0,5 px ao INT8. Não houve nenhum unmatched
na configuração final. Na tentativa performance, `webcam_00` teve
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
`cpu/test_common.py` passou em oito testes, incluindo permutação, erros de
caixa/score/classe/contagem, conjuntos vazios, fronteira unmatched 0,30,
inversão de letterbox com filtro 0,001 e as três regressões da revisão:
IoU ≈ 0,05 com scores 0,26 passa como dois unmatched; com scores 0,40 falha;
IoU ≈ 0,85 com classes diferentes falha como par. A validação INT8 isolada
e a completa foram repetidas: 8/8 e 40/40 PASS, respectivamente.

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

## Benchmark controlado das cinco variantes

### Controle térmico

A revisão identificou deriva térmica na sequência anterior: frequência média
4,21 → 3,90 → 3,71 → 3,36 → 3,09 GHz, na ordem eager, ORT, OV FP32,
OV INT8, OV default. A execução anterior não tinha outro trabalho pesado
detectado, mas isso **não bastava para controlar a temperatura**. Os testes
isolados do revisor, após 45 s de cooldown, confirmaram a penalização das
variantes posteriores. Números fornecidos pelo revisor, não reexecutados aqui:

| Variante | Sequencial anterior: média ms / GHz | Revisor, isolado com 45 s: média ms / GHz | Final, três rodadas com 60 s: medianas ms / GHz |
|---|---:|---:|---:|
| ort-cpu FP32 | 46,43 / 3,90 | 43,57 / 4,14 | 43,63 / 4,125 |
| ov-cpu FP32 | 43,43 / 3,71 | 41,16 / 4,01 | 40,20 / 4,036 |
| ov-cpu INT8 | 19,35 / 3,36 | 15,17 / 4,19 | 14,91 / 4,279 |

O script agora espera **60 s antes de cada tentativa**, faz três rodadas e
rotaciona a ordem por deslocamentos 0, 2 e 4:

1. eager FP32 → ORT FP32 → OV FP32 → OV INT8 → OV FP32 default;
2. OV FP32 → OV INT8 → OV FP32 default → eager FP32 → ORT FP32;
3. OV FP32 default → eager FP32 → ORT FP32 → OV FP32 → OV INT8.

Cada variante continua em processo independente, com 20 aquecimentos + 200
medidas e as mesmas propriedades/afinidades. Após o cooldown, o check de cinco
segundos verifica o host; o monitor de 500 ms acompanha a execução. O monitor
**não foi alterado**, incluindo o limiar agregado de 75% de uma CPU lógica e
a lista de compiladores. Descartes recebem código 75: o script espera novos
60 s e retenta até cinco vezes além da tentativa inicial; ao esgotar, falha
explicitamente. Erros de modelo/configuração não são retentados. Arquivos de
resultados e resumo antigos são removidos no início, impedindo resumo obsoleto.

**O benchmark final foi executado pelo orquestrador em uma janela tranquila,
com o navegador fechado segundo seu relato e os containers Docker de outros
projetos ainda rodando, por decisão do usuário.** Os containers não foram
parados nem modificados. O monitor permaneceu inalterado; não houve download,
calibração ou avaliação COCO em paralelo. O log registra tentativas descartadas
antes de obter os 15 resultados aceitos, conforme detalhado abaixo.

### Resultado final: medianas das três rodadas

A tentativa inicial do agente esgotou as retentativas por concorrência. Depois,
o orquestrador completou as três rodadas: **15 execuções aceitas, três por
variante**. Esta atualização apenas leu os artefatos existentes, sem reexecutar
o benchmark. Os blocos estão em `artifacts/bench/round3.txt`, os JSONs por rodada
em `round3-runs/` e o resumo em `round3-summary.txt`.

**Ressalva de proveniência:** embora o relato da execução tenha dito que não
houve descartes, o log efetivamente contém **seis tentativas descartadas**:
uma de OV default na rodada 1; três de eager, uma de ORT e uma de OV FP32 na
rodada 3. Todas foram retentadas com novo cooldown e não entram nas medianas.
Os 15 blocos aceitos e o resumo final estão completos; não houve relaxamento
do monitor nem mistura com a sequência antiga.

Cada mediana usa **três valores, um por execução**, sem juntar as 600 iterações;
min/max referem-se às três latências médias. RSS, carga, primeiro forward e
tamanho abaixo também são medianas dos três blocos aceitos de `round3.txt`.
FPS é a mediana dos throughputs por execução, não FPS de câmera.

| Variante | Mediana das médias ms | Mediana p95 ms | Mediana FPS | Mediana P-core GHz | Min/max das médias ms | RSS pico MiB | Carga ms | Primeiro ms | Modelo MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eager-cpu FP32 | 55,56 | 64,51 | 18,0 | 4,344 | 53,92–55,69 | 929,4 | 408,26 | 64,45 | 15,853 |
| ort-cpu FP32 | 43,63 | 49,96 | 22,9 | 4,125 | 43,36–44,17 | 218,5 | 79,61 | 43,39 | 28,358 |
| ov-cpu FP32 | 40,20 | 45,73 | 24,9 | 4,036 | 39,88–40,42 | 214,9 | 130,25 | 37,70 | 28,021 |
| ov-cpu INT8 | 14,91 | 16,04 | 67,1 | 4,279 | 14,82–15,23 | 171,9 | 272,90 | 15,39 | 7,722 |
| ov-cpu FP32 default | 39,84 | 41,12 | 25,1 | 4,035 | 39,62–40,13 | 214,4 | 132,55 | 38,16 | 28,021 |

Para a comparação com o teto, usam-se **as medianas impressas acima** e
21,703 GFLOPs (arredondamento dos 21,7026048 verificados no modelo fundido):
`pico = 6 × 32 × GHz`, `atingido = 21,703 × 1000 / latência_ms` e
`percentual = atingido / pico × 100`. Portanto não se toma a mediana dos
percentuais de cada execução. Os speedups também usam as latências medianas.

| Variante | Pico FP32 GFLOPS | GFLOPS atingidos / equivalentes | % do pico FP32 | Speedup vs eager CPU FP32 |
|---|---:|---:|---:|---:|
| eager-cpu FP32 | 834,05 | 390,62 | 46,83% | 1,00× |
| ort-cpu FP32 | 792,00 | 497,43 | 62,81% | 1,27× |
| ov-cpu FP32 | 774,91 | 539,88 | 69,67% | 1,38× |
| ov-cpu INT8 | 821,57 | 1455,60 equivalentes | **177,17% equivalente, não utilização FP32** | 3,73× |
| ov-cpu FP32 default | 774,72 | 544,75 | 70,32% | 1,39× |

O INT8 entrega **throughput FP32-equivalente**, pois substitui a maior parte
da aritmética por operações quantizadas: 177,17% não é utilização das unidades
FP32 nem do teto INT8. A comparação direta INT8/OV FP32 explícito é **2,70×**,
ou **62,9% menos latência**, mantendo a queda de mAP50–95 em 0,3094 ponto.

## Comparação com as rodadas GPU e conclusão

Números **citados, não reexecutados**, de
[`tensorrt-results.md`](tensorrt-results.md), RTX 3050 Laptop:

| Runtime GPU | FP32 ms | FP16 ms |
|---|---:|---:|
| eager Python | 9,86 | 5,59 |
| AOTInductor C++ | 8,54 | 3,89 |
| TensorRT C++ padrão | 7,44 (**1 FAIL estrito**) | 2,95 |
| TensorRT C++ sem TF32 | 7,87 (8/8 PASS) | — |

A GPU mantém vantagem: o melhor FP32 CPU, OV default a 39,84 ms, é **4,04×
mais lento que eager GPU FP32** (9,86 ms), **4,67× que AOTI FP32** (8,54 ms)
e **5,06× que TRT FP32 sem TF32** (7,87 ms). INT8 CPU a 14,91 ms ainda é
**5,05× mais lento que TRT GPU FP16** (2,95 ms). São precisões e protocolos
diferentes: GPU usa 50+500 iterações em uma sequência; CPU, 20+200 por execução
e medianas de três rodadas com cooldown. Não é um teste de hardware em
condições idênticas, e o TRT FP32 padrão com FAIL estrito não é uma referência
de substituição numericamente validada.

**OV default não é mais lento que a configuração manual de seis P-cores**:
39,84 contra 40,20 ms, diferença de apenas 0,36 ms / 0,9%, com faixas de
latência média sobrepostas. Seu p95 mediano é melhor: **41,12 contra 45,73 ms**
(10,1% menor). As frequências medianas são praticamente iguais, 4,035 e
4,036 GHz. A antiga conclusão de **27,1% de penalidade era um artefato térmico
de executar o default por último**, não uma demonstração de vantagem da
fixação manual. A diferença pequena das medianas atuais também não demonstra
superioridade universal do default; são três execuções nesta máquina/versão.

Para um futuro engine C++ próprio, o alvo FP32 fica em **aproximadamente
40 ms** (39,84 ms default; 40,20 ms explícito), e o alvo INT8 em **14,91 ms**.
OV FP32 já entrega **cerca de 70% do pico FP32 estimado**: 544,75 de
774,72 GFLOPS no default, ou 539,88 de 774,91 no explícito. Isso deixa pouca
margem para um engine FP32 escrito à mão, sobretudo se a proposta for apenas
retirar Python. No default, o limite puramente aritmético seria **28,01 ms**,
no máximo ideal **1,42× / 29,7% de redução** até 100%; não é promessa atingível.
Memória, ativações, atenção, TopK, conversões e escalonamento não são só FMAs
densas e impedem tratar toda essa margem como recuperável.

**INT8 é a principal alavanca:** **2,70× sobre OV FP32 explícito**, redução de
62,9% da latência, e **3,73× sobre eager CPU FP32**, com apenas **0,31 ponto
de queda de mAP50–95** nas 500 imagens (limite 1,0 ponto). Essa troca de
precisão oferece ganho maior que a margem aritmética ideal restante do FP32.
O percentual INT8 contra o pico FP32 é apenas equivalente; **não se deduz a
distância ao teto INT8** a partir dele.

Limitações: notebook híbrido com boost e térmica variáveis, sem fixar clocks,
potência, governador ou confirmar alimentação pela tomada. O novo protocolo
usa ordem rotativa determinística, não aleatória, três repetições e faixa
mínima/máxima, não intervalos de confiança. Cooldown de 60 s reduz a deriva,
mas não garante temperatura inicial idêntica. As frequências continuam medidas,
não são uma constante nominal.
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
mkdir -p artifacts  # MAIN_CHECKOUT: checkout principal com os artefatos da rodada 1
cp -r "$MAIN_CHECKOUT"/artifacts/{data,ref,yolov10s_fp32.onnx} artifacts/
cp "$MAIN_CHECKOUT"/yolov10s.pt .
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
bash scripts/bench_cpu.sh

uv run python cpu/test_common.py
uv run python cpu/test_bench.py
uv run python -m py_compile cpu/*.py
bash -n scripts/bench_cpu.sh
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

Verificação original de acurácia: `uv sync`, quantização e mAP completo OK,
com queda de 0,3094 ponto; referências e IRs preservados nesta revisão.
Verificação repetida na revisão: validação INT8 isolada **8 PASS / 0 FAIL**;
validação completa **40 PASS / 0 FAIL**; oito testes de validação e quatro do
protocolo/resumo OK; compilação sintática Python, sintaxe Bash e
`git diff --check` OK. Os testes do benchmark verificam medianas por execução,
faixa min/max, ordem 0/2/4, cooldown de cada tentativa, retentativa somente
para concorrência, limite de cinco retentativas e erro comum sem retentativa.
A tentativa anterior do agente falhou por concorrência. **Execução final pelo
orquestrador: 15 resultados aceitos**, com três por variante e resumo completo;
o log preserva seis descartes/retentativas, excluídos da tabela. Nesta atualização
somente documental, os 15 blocos e as medianas foram conferidos contra
`artifacts/bench/round3.txt` e `round3-summary.txt`, inclusive RSS/carga/primeiro
forward/tamanho e cálculos de pico/speedup. O benchmark não foi reexecutado.
As conclusões termicamente confundidas foram substituídas pelos resultados
controlados, sem alterar código, tolerâncias, referências ou artefatos.
