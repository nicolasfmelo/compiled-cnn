# PBI-099 — Ajuste do TensorRT FP16 na GPU

## Objetivo

Medir separadamente o nível do builder, CUDA Graph e a entrada retangular da YOLOv10s, sem mudar pesos ou precisão. A comparação retangular × quadrada é um experimento de detecção, não a validação dos engines contra eager.

## Ambiente e versões

Sessão de 16/11/2026 no mesmo notebook das rodadas anteriores: RTX 3050 6GB Laptop, driver 620.70, TensorRT 11.4.0.106 e CUDA 13.4. Clocks não travados; os planos foram reconstruídos pelo coordenador com a GPU ociosa. Engines FP16 estáticos, `--stronglyTyped`, batch 1. A entrada webcam é 640×384, tensor NCHW `1×3×384×640`.

## O que mudou no código

- `trt/export_rect.py`: exporta ONNX FP16 e referências eager separadas para 640×384.
- `trt/build_tuned.sh`: constrói cinco planos independentes, sem compartilhar cache de timing.
- `trt/profile_tuned.sh`: perfil de A e D com `trtexec`; se necessário, usa build detalhado separado.
- `trt/profile_top.py`: lista as dez camadas de maior média do JSON de perfil.
- `scripts/bench_trt_tuning.sh`: executa variantes em processos separados e ordem rotativa.
- `scripts/summarize_trt_tuning.py`: resume medianas e speedups dos arquivos por rodada.
- `trt/eager_rect_scores.py`: lê scores eager, inclusive abaixo do limiar.
- `native/common.h`: letterbox e validação aceitam o formato de entrada, mantendo o padrão quadrado.
- `aoti/export.py`: pré-processamento e coordenadas aceitam o formato escolhido.
- `aoti/bench_eager.py --height 384`: mede eager na altura 384, largura 640.
- `yolo_trt validate --graph`: valida com captura do lançamento; `bench --graph` mede o grafo.
- `yolo_trt bench --ref-dir` e `--label`: escolhem referência e rótulo; eventos CUDA medem tempo na GPU.
- `yolo_trt compare`: compara dois planos nas coordenadas originais, sem opção de grafo.

## Comandos de reprodução

Partindo das referências e ONNX da rodada 2 e do runner compilado (veja [guia](REPRODUCE.md#9-tensorrt-fp16-tuning-pbi-099)):

```bash
uv run python trt/export_rect.py
bash trt/build_tuned.sh
B=build/bin/yolo_trt
T=artifacts/trt_tuning
$B validate $T/fp16_640_o3_b1.plan artifacts/ref
$B validate $T/fp16_640_o3_b2.plan artifacts/ref
$B validate $T/fp16_640_o5_b1.plan artifacts/ref
$B validate $T/fp16_640_o5_b2.plan artifacts/ref
$B validate $T/fp16_640_o5_b1.plan artifacts/ref --graph
$B validate $T/fp16_384x640_o5.plan artifacts/ref_384x640 --graph
$B detect $T/fp16_384x640_o5.plan artifacts/data/bus.jpg --check artifacts/ref_384x640/bus.eager_fp16_dets.txt
$B detect $T/fp16_384x640_o5.plan artifacts/data/zidane.jpg --check artifacts/ref_384x640/zidane.eager_fp16_dets.txt
$B compare $T/fp16_640_o5_b1.plan $T/fp16_384x640_o5.plan artifacts/data/zidane.jpg artifacts/data/webcam_0{0..5}.jpg
uv run python trt/eager_rect_scores.py
bash trt/profile_tuned.sh
bash scripts/bench_trt_tuning.sh
```

`compare` retorna falha nesta amostra, como detalhado abaixo; não interrompa a sequência por isso. Ao carregar dois engines, o TensorRT avisa que o segundo logger é ignorado: aviso inofensivo. A carga com `--graph` inclui um enqueue de aquecimento e a captura.

## Builds

A: 640×640, nível padrão 3 e `enqueueV3`; B: 640×640, nível 5 e `enqueueV3`; C: plano B-b1 com CUDA Graph; D: 640×384, nível 5 e grafo. A e B têm dois builds independentes.

| Plano | Build (s) | Tamanho (MiB) |
|---|---:|---:|
| A-b1 | 63,9 | 81,5 |
| A-b2 | 62,9 | 81,4 |
| B-b1 | 204,4 | 82,5 |
| B-b2 | 214,0 | 82,3 |
| D | 240,6 | 56,3 |

Os tempos estão em `artifacts/trt_tuning/build_*.seconds`; o nível 5 custou cerca de 3,3× o build do nível 3. C reutiliza B-b1.

## Validação

A-b1, A-b2, B-b1, B-b2 e C passaram 8/8 contra `artifacts/ref/`; D passou 8/8 contra `artifacts/ref_384x640/`. As tolerâncias FP16 não mudaram: caixas ≤ 2 px, scores ≤ 5e-3 e pares por IoU ≥ 0,95, com mesma contagem e classe acima de 0,25. Em 640×384, `detect --check` passou em bus 5/5 (IoU mínimo 0,9992) e zidane 3/3 (0,9995). Um plano 640×640 rejeita referências retangulares por tamanho incorreto. Os formatos dos runners da rodada 2 permanecem: `yolov10s_fp16.plan` 8/8, `yolo_aoti` 8/8 e ctest 5/5. Logs atuais: `artifacts/trt_tuning/validate_*.log` e `detect_*.log`; não usar `stale_first_build/`.

## Comparação retangular × quadrada

Nas sete imagens 16:9, `compare` usa B-b1 com `enqueueV3` simples e D; B-b1 é o mesmo plano de C e as saídas não dependem do grafo (as linhas de validação de B-b1 e C são idênticas). Critério fixado: mesma contagem e classe, IoU ≥ 0,90 e diferença de score ≤ 0,02. Bus, em retrato, ficou fora.

| Imagem | Resultado | Detecções retangular / quadrada | Observação |
|---|---|---|---|
| zidane | FAIL | 3/3 | IoU mínimo 0,9899; diferença máxima 0,0254 na gravata: 0,832 → 0,807; pessoas 0,965 → 0,963 e 0,922 → 0,918 |
| webcam_00 | PASS | 0/0 | sem detecções |
| webcam_01 | PASS | 0/0 | sem detecções |
| webcam_02 | PASS | 0/0 | sem detecções |
| webcam_03 | FAIL | 0/1 | quadrada: classe 59, score 0,263 |
| webcam_04 | FAIL | 0/1 | quadrada: classe 59, score 0,269 |
| webcam_05 | FAIL | 0/1 | quadrada: classe 61, score 0,269 |

As três detecções exclusivas da quadrada tocam a borda inferior (y2 ≈ 711–718 em quadros de 720 px). O critério não foi afrouxado. Scores eager FP16 máximos por classe, inclusive abaixo de 0,25 (`artifacts/trt_tuning/eager_rect_scores.txt`):

| Imagem / classe | 640×640 | 640×384 |
|---|---:|---:|
| webcam_03 / 59 | 0,2451 | 0,1656 |
| webcam_04 / 59 | 0,2673 | 0,1490 |
| webcam_05 / 61 | 0,2830 | 0,2379 |
| zidane / gravata | 0,8320 | 0,8066 |

A queda também ocorre no PyTorch: é do modelo, não do TensorRT. Em webcam_03, eager quadrado já está abaixo de 0,25 (0,245), portanto tem 0 detecções nos dois formatos. A detecção quadrada de 0,263 no `compare` vem do letterbox C++ (stb_image e resize bilinear do JPEG), ligeiramente diferente do pré-processamento Python/OpenCV das referências, deslocando scores limítrofes em até cerca de 0,02 (webcam_05: 0,269 em C++ contra 0,283 eager). Com 12 px em vez de 140 px de padding cinza acima e abaixo, objetos que tocam as bordas superior/inferior perdem contexto e os de baixa confiança podem cair abaixo de 0,25; detecções fortes mantêm caixa e classe (IoU ≥ 0,989).

## Benchmark

Um processo por variante, três rodadas com ordem rotativa, pausa de 30 s antes de cada execução, entrada já na GPU, 50 aquecimentos e 500 iterações medidas. A tabela mostra a mediana das três rodadas para cada estatística. Host é relógio de parede do lançamento até a sincronização do stream. GPU usa eventos CUDA ao redor do lançamento: tempo do primeiro ao último trabalho no dispositivo; com `enqueueV3` simples inclui intervalos ociosos enquanto a CPU ainda lança kernels. Portanto B→C também mostra o custo de lançamento na linha do tempo da GPU. O perfil por camada isola o tempo dos kernels com overhead de perfil.

| Variante | Host média | Host p50 | Host p95 | GPU média | GPU p50 | GPU p95 | vs A-b1 | vs eager mesmo tamanho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eager-640 | 5,61 | 5,48 | 6,34 | — | — | — | 0,53× | 1,00× |
| eager-384 | 3,61 | 3,53 | 4,34 | — | — | — | 0,83× | 1,00× |
| A-o3-b1 | 2,98 | 2,92 | 3,36 | 2,96 | 2,91 | 3,02 | 1,00× | 1,88× |
| A-o3-b2 | 2,97 | 2,93 | 3,13 | 2,96 | 2,93 | 3,09 | 1,00× | 1,89× |
| B-o5-b1 | 2,94 | 2,88 | 3,27 | 2,91 | 2,87 | 3,02 | 1,01× | 1,91× |
| B-o5-b2 | 2,92 | 2,86 | 3,68 | 2,91 | 2,85 | 3,25 | 1,02× | 1,92× |
| C-o5-b1-graph | 2,78 | 2,67 | 3,26 | 2,72 | 2,66 | 3,04 | 1,07× | 2,02× |
| `D-384x640-o5-graph` | 1,69 | 1,62 | 1,86 | 1,63 | 1,61 | 1,75 | 1,76× | 2,14× |

Tempos em ms; FPS = 1000 / média host: A-b1 336, C 360, D 592. Resumo em `artifacts/trt_tuning/bench/summary.txt`, medições individuais em `round-<r>-<variant>.txt` no mesmo diretório.

Nível 5 versus 3: 2,94/2,92 contra 2,98/2,97 ms, diferença de 1–2%, indistinguível do ruído. A variação entre builds A foi 0,01 ms na média (p95 varia mais). A reproduz a rodada 2: 2,98 contra 2,94 ms, diferença de 1,4%, dentro de ±3%. No smoke de uma rodada do executor, A-b1 e B-b1 mediram ambos 2,95 ms. Não há ganho mensurável do nível 5 apesar do build mais longo.

B→C reduz host de 2,94 para 2,78 ms (−5%) e GPU de 2,91 para 2,72 ms: parte do tempo sem grafo é espera pelos lançamentos. Corroboração: o timing do `trtexec` com CUDA Graph no plano A-b1 deu GPU média 2,74 ms (mediana 2,70), próximo dos 2,72 ms de C com nível 5; o ganho A→C vem do grafo. D contra C: 1,69 contra 2,78 ms (1,64×). A razão de GPU 1,63/2,72 = 0,60 coincide com a razão de pixels 384/640 = 0,60 nesta medição, sem implicar lei linear geral. A→D: 1,76×; D contra eager no mesmo tamanho: 2,14×.

## Perfil por camada

`trtexec --dumpProfile --separateProfileRun` nos planos medidos A-b1 e D; dados em `artifacts/trt_tuning/profile_session.log` e `profile_*.json`. A soma das médias é 3,20 ms em A e 2,08 ms em D (0,65×). A sincronização por camada introduz overhead, e a soma de A supera seus 2,96 ms de tempo GPU. O perfil é plano: a camada mais cara representa 2,65% em A e 2,90% em D. O TopK do head end-to-end custa 0,060 ms em ambos e não diminui com a entrada, ajudando a explicar a razão por camada maior que a razão de pixels.

### A-b1: dez maiores

| Posição | Média (ms) | % | Camada |
|---:|---:|---:|---|
| 1 | 0,084992 | 2,65% | `__myl_CastMulTanhMulAddMulCast_myl3_0` |
| 2 | 0,081359 | 2,54% | `node_conv2d_4 + PWN(PWN(PWN(node_Sigmoid_6), PWN(node_silu_4)), PWN(node_add))` |
| 3 | 0,078601 | 2,45% | `node_conv2d_1 + PWN(PWN(node_Sigmoid_1), PWN(node_silu_1))` |
| 4 | 0,068054 | 2,13% | `node_conv2d` |
| 5 | 0,063633 | 1,99% | `node_conv2d_6 + PWN(PWN(node_Sigmoid_9), PWN(node_silu_6))` |
| 6 | 0,060416 | 1,89% | `__myl_Topk_myl194_3` |
| 7 | 0,057247 | 1,79% | `node_conv2d_43 + PWN(PWN(node_Sigmoid_67), PWN(node_silu_37))` |
| 8 | 0,057185 | 1,79% | `node_conv2d_5 + PWN(PWN(node_Sigmoid_8), PWN(node_silu_5))` |
| 9 | 0,056643 | 1,77% | `node_conv2d_61 + PWN(PWN(node_Sigmoid_84), PWN(node_silu_54))` |
| 10 | 0,049356 | 1,54% | `node_conv2d_13 + PWN(PWN(node_Sigmoid_17), PWN(node_silu_13))` |

### D: dez maiores

| Posição | Média (ms) | % | Camada |
|---:|---:|---:|---|
| 1 | 0,060416 | 2,90% | `__myl_Topk_myl204_3` |
| 2 | 0,052397 | 2,52% | `node_conv2d_1 + PWN(PWN(node_Sigmoid_1), PWN(node_silu_1))` |
| 3 | 0,052223 | 2,51% | `__myl_CastMulTanhMulAddMulCast_myl3_0` |
| 4 | 0,042401 | 2,04% | `node_conv2d` |
| 5 | 0,041550 | 2,00% | `node_conv2d_6 + PWN(PWN(node_Sigmoid_9), PWN(node_silu_6))` |
| 6 | 0,036707 | 1,76% | `node_conv2d_5 + PWN(PWN(node_Sigmoid_8), PWN(node_silu_5))` |
| 7 | 0,035849 | 1,72% | `node_conv2d_43 + PWN(PWN(node_Sigmoid_67), PWN(node_silu_37))` |
| 8 | 0,034696 | 1,67% | `node_conv2d_61 + PWN(PWN(node_Sigmoid_84), PWN(node_silu_54))` |
| 9 | 0,031423 | 1,51% | `node_conv2d_4 + PWN(PWN(PWN(node_Sigmoid_6), PWN(node_silu_4)), PWN(node_add))` |
| 10 | 0,031279 | 1,50% | `node_conv2d_13 + PWN(PWN(node_Sigmoid_17), PWN(node_silu_13))` |

## Conclusão

CUDA Graph traz ganho mensurável sem mudar a entrada; nível 5 não mostrou ganho distinguível do ruído. O perfil webcam reduz a latência, mas altera a carga e pode perder detecções limítrofes nas bordas. Todos os engines passaram contra eager no próprio tamanho; isso não significa equivalência entre os dois formatos.

## Limitações

Um único notebook, clocks destravados e apenas oito imagens de validação; não se trata de avaliação de acurácia em dataset. A comparação retangular cobre apenas os sete quadros 16:9 desta amostra. O perfil por camada tem overhead de sincronização e não é diretamente comparável ao tempo GPU da inferência completa.
