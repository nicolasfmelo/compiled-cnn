"""Gera o relatório final do PBI-095 em HTML (autocontido) e PDF, a partir da mesma fonte.

Uso (na raiz do repositório):
    uv run --no-project --with weasyprint --with matplotlib python docs/report/build_report.py

Os números abaixo vêm dos relatórios técnicos de cada rodada (docs/*-results.md) e dos
arquivos de benchmark em artifacts/bench/. Atualize-os aqui depois de rodar os benchmarks de novo.
"""
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
HTML_OUT = HERE / "relatorio-pbi-095.html"
PDF_OUT = HERE / "relatorio-pbi-095.pdf"

# Paleta categórica (slots 1-4 validados para barras adjacentes) e tinta de texto.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#1f1f1e", "#4a4a48", "#6f6f6c", "#e4e4e1"


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    text = buf.getvalue()
    return text[text.index("<svg"):]


def style_axes(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0, colors=INK2)
    ax.tick_params(axis="x", colors=MUTED)


def chart_gpu():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"})
    groups = ["FP32", "FP16"]
    series = [("PyTorch eager (Python)", BLUE, [100, 179]),
              ("AOTInductor (C++)", ORANGE, [118, 255]),
              ("TensorRT (C++)", AQUA, [135, 340]),
              ("ncnn Vulkan (C++)", YELLOW, [54, 103])]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    h, gap = 0.19, 0.02
    for gi in range(len(groups)):
        for si, (name, color, vals) in enumerate(series):
            y = gi + (si - 1.5) * (h + gap)
            ax.barh(y, vals[gi], height=h, color=color, label=name if gi == 0 else None, zorder=3)
            ax.text(vals[gi] + 4, y, f"{vals[gi]}", va="center", ha="left", color=INK, fontsize=8.5)
    ax.set_yticks(range(len(groups)), groups)
    ax.invert_yaxis()
    ax.set_xlim(0, 400)
    ax.set_xlabel("Inferências por segundo (só o modelo, 640×640, RTX 3050)", color=INK2)
    style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2)
    return svg(fig)


def chart_cpu():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"})
    rows = [("PyTorch eager FP32", 18.0), ("ncnn FP32", 17.6), ("ONNX Runtime FP32", 22.9),
            ("OpenVINO FP32", 24.9), ("OpenVINO INT8", 67.1)]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for i, (name, fps) in enumerate(rows):
        ax.barh(i, fps, height=0.6, color=BLUE, zorder=3)
        ax.text(fps + 0.8, i, f"{fps:.1f}".replace(".", ","), va="center", ha="left", color=INK,
                fontsize=8.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 80)
    ax.set_xlabel("Inferências por segundo (i5-13450HX, 6 threads nos P-cores; medianas)", color=INK2)
    style_axes(ax)
    return svg(fig)


CSS = """
:root { --ink:#1f1f1e; --ink2:#4a4a48; --muted:#6f6f6c; --line:#e4e4e1; --zebra:#f6f6f4;
        --panel:#f1f5fb; --accent:#2a78d6; --bg:#ffffff; --warn:#b5520f; }
@media screen and (prefers-color-scheme: dark) {
  :root { --ink:#ededea; --ink2:#c9c9c5; --muted:#a3a39f; --line:#3a3a38; --zebra:#232322;
          --panel:#1d2633; --accent:#5b9cf0; --bg:#161615; --warn:#f0a26a; }
  svg text { fill: var(--ink2) !important; }
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body { margin: 0 auto; max-width: 900px; padding: 32px 16px 64px; color: var(--ink);
       background: var(--bg); font: 15px/1.6 "DejaVu Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 30px; line-height: 1.2; margin: 0 0 6px; }
h2 { font-size: 21px; margin: 40px 0 10px; padding-top: 6px; border-top: 1px solid var(--line); }
h3 { font-size: 16px; margin: 22px 0 6px; }
p, li { color: var(--ink); }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 18px; }
.small { color: var(--muted); font-size: 12.5px; line-height: 1.5; }
code { font-family: "DejaVu Sans Mono", ui-monospace, monospace; font-size: 0.88em;
       background: var(--zebra); padding: 1px 4px; border-radius: 4px; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 18px 0 8px; }
.kpi { background: var(--panel); border-radius: 10px; padding: 12px 14px; }
.kpi b { display: block; font-size: 22px; line-height: 1.2; }
.kpi span { color: var(--muted); font-size: 12.5px; line-height: 1.35; display: block; margin-top: 4px; }
.table-wrap { overflow-x: auto; margin: 8px 0 6px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; font-weight: 700; border-bottom: 1.5px solid var(--ink2); padding: 6px 7px; vertical-align: bottom; }
td { border-bottom: 1px solid var(--line); padding: 5px 7px; vertical-align: top; }
tbody tr:nth-child(even) td { background: var(--zebra); }
td.num, th.num { text-align: right; white-space: nowrap; }
tr.best td { font-weight: 700; }
.figure { margin: 10px 0 4px; }
.figure svg { width: 100%; height: auto; }
.note { border-left: 3px solid var(--accent); background: var(--panel); padding: 10px 14px;
        border-radius: 0 8px 8px 0; margin: 12px 0; }
.warn { color: var(--warn); font-weight: 700; }
nav ol { columns: 2; font-size: 14px; padding-left: 20px; }
@media (max-width: 640px) { .kpis { grid-template-columns: repeat(2, 1fr); } nav ol { columns: 1; } h1 { font-size: 24px; } }
@page { size: A4; margin: 17mm 16mm 18mm;
        @bottom-left { content: "PBI-095 · YOLOv10s compilada para runtimes nativos"; font: 8pt "DejaVu Sans"; color: #6f6f6c; }
        @bottom-right { content: "Página " counter(page) " de " counter(pages); font: 8pt "DejaVu Sans"; color: #6f6f6c; } }
@media print {
  body { max-width: none; padding: 0; font-size: 9.6pt; line-height: 1.5; }
  h1 { font-size: 21pt; } h2 { font-size: 14pt; margin-top: 22px; break-after: avoid; }
  h3 { font-size: 11pt; break-after: avoid; }
  table { font-size: 8pt; } .kpi b { font-size: 15pt; } .kpi span { font-size: 8pt; }
  .table-wrap, .figure, .kpis, .note { break-inside: avoid; }
  nav { display: none; }
  h2.newpage { break-before: page; border-top: none; }
}
"""


def table(header, rows, num_cols=(), best=(), cls=""):
    th = "".join(f'<th class="num">{h}</th>' if i in num_cols else f"<th>{h}</th>"
                 for i, h in enumerate(header))
    body = []
    for r, row in enumerate(rows):
        tds = "".join(f'<td class="num">{c}</td>' if i in num_cols else f"<td>{c}</td>"
                      for i, c in enumerate(row))
        body.append(f'<tr class="best">{tds}</tr>' if r in best else f"<tr>{tds}</tr>")
    return (f'<div class="table-wrap {cls}"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def build_html():
    gpu_chart, cpu_chart = chart_gpu(), chart_cpu()
    S = []
    S.append("""
<h1>YOLOv10s compilada para runtimes nativos</h1>
<div class="sub">PBI-095 · Relatório completo do experimento · 12 de novembro de 2026 · notebook com RTX 3050 6GB Laptop e i5-13450HX</div>

<p>Este relatório reúne as quatro rodadas de um experimento sobre uma pergunta simples: <b>quanto se ganha ao compilar uma rede de detecção para um artefato nativo e executá-la a partir de C++</b>, em vez de rodar o PyTorch em Python, o que esse artefato ainda exige para funcionar e se o resultado numérico continua o mesmo. A rede é a YOLOv10s (Ultralytics). Foram testados AOTInductor e TensorRT na GPU NVIDIA, ONNX Runtime e OpenVINO (FP32 e INT8) na CPU, e ncnn via Vulkan na GPU NVIDIA, na GPU integrada Intel e na CPU. Todos os backends foram validados contra a mesma referência, com tolerâncias fixas, e medidos com o mesmo protocolo.</p>

<div class="kpis">
  <div class="kpi"><b>2,94 ms</b><span>TensorRT FP16 na RTX 3050: 340 FPS, 1,90× o PyTorch eager</span></div>
  <div class="kpi"><b>14,91 ms</b><span>OpenVINO INT8 na CPU: 67 FPS, 3,73× o eager, com −0,31 ponto de mAP</span></div>
  <div class="kpi"><b>117 de 120</b><span>validações PASS; as 3 exceções têm causa conhecida (TF32 e FP16 na GPU integrada)</span></div>
  <div class="kpi"><b>sem Python</b><span>três runners C++ (AOTInductor, TensorRT e ncnn) executam a rede sem interpretador</span></div>
</div>

<nav><ol>
<li>Motivação</li><li>Hardware</li><li>Método comum</li><li>Ponto de partida: a webcam</li>
<li>Rodada 1: AOTInductor</li><li>Rodada 2: TensorRT</li><li>Rodada 3: CPU</li><li>Rodada 4: ncnn e Vulkan</li>
<li>Comparação geral</li><li>Problemas e lições</li><li>Conclusão</li><li>Apêndice: versões e reprodução</li>
</ol></nav>
""")

    # 1. Motivação
    S.append("""
<h2>1. Motivação</h2>
<p>A ideia de partida, registrada no card PBI-095, é tratar uma rede neural pelo que ela é na hora da inferência: <b>um grafo de operações mais um conjunto de pesos</b>. Em vez de depender de Python e do PyTorch para rodar, esse grafo pode ser compilado para uma biblioteca ou um executável nativo. O interesse é prático: entender a viabilidade, as dependências que sobram e os ganhos reais antes de integrar uma solução à linguagem Enty (PBI-096). O card deixa claro um cuidado: compilar ou embutir os pesos <i>não garante</i> mais velocidade, então o ganho precisa ser medido, inclusive quando não houver.</p>
<p>O experimento responde a três perguntas:</p>
<ul>
<li><b>Viabilidade:</b> o modelo exporta e compila sem operações customizadas, e o resultado numérico continua o mesmo?</li>
<li><b>Dependências:</b> o que o artefato ainda exige para rodar (libtorch, CUDA, drivers, runtimes)? Os pesos ficam embutidos ou externos?</li>
<li><b>Ganho:</b> quanto se ganha em latência, vazão e memória em relação ao PyTorch eager, nas mesmas condições?</li>
</ul>
<p>O card sugeria começar pela CPU e pela YOLOv8m. As decisões tomadas ao longo do caminho foram: começar pela <b>GPU</b>, que é onde a webcam roda; usar a <b>YOLOv10s</b>, caso de interesse do card e que tem uma vantagem para comparar backends (o head end2end entrega as detecções finais <b>sem NMS</b>, então todo backend faz exatamente o mesmo trabalho); e depois cobrir a CPU e a GPU via Vulkan.</p>
""")

    # 2. Hardware
    S.append("<h2>2. Hardware</h2>")
    S.append("<p>Os resultados dependem muito do que o hardware sabe ou não sabe fazer, então vale descrever a máquina em detalhe.</p>")
    S.append(table(["Componente", "Detalhes"], [
        ["<b>GPU</b>", "NVIDIA GeForce RTX 3050 6GB Laptop: chip GA107, arquitetura Ampere, capacidade 8.6, 20 SMs (2.560 núcleos CUDA e 80 tensor cores de 3ª geração), 6 GB GDDR6, até 2.100 MHz, limite de 95 W. Driver 620.70."],
        ["<b>CPU</b>", "Intel Core i5-13450HX (Raptor Lake, híbrida): <b>6 P-cores</b> com Hyper-Threading (CPUs lógicas 0–11, até 4,6 GHz) e <b>4 E-cores</b> (CPUs 12–15, até 3,4 GHz), 16 threads, 9,5 MiB de L2 e 20 MiB de L3. Instruções: AVX2, FMA, F16C e <b>AVX-VNNI</b>. Não tem AVX-512, AVX-512 FP16 nem AMX."],
        ["<b>GPU integrada</b>", "Intel UHD Graphics (Raptor Lake-S, PCI 8086:A78B), até 1.450 MHz, driver Mesa 26.2 (Vulkan 1.4). Suporta FP16; não tem cooperative matrix."],
        ["<b>Memória</b>", "16 GB (15 GiB visíveis)."],
        ["<b>Sistema</b>", "CachyOS (base Arch), Linux 7.3, GCC 16.2, CMake 4.4. CUDA 13.4 do sistema (usado pelo TensorRT) e CUDA 12.8 das wheels do PyTorch (usado pelo eager e pelo AOTInductor)."],
    ]))
    S.append("""
<h3>Por que isso importa</h3>
<ul>
<li><b>Tensor cores</b> (FP16 e TF32) explicam por que o FP16 quase dobra a vazão na GPU, e por que o TF32 que o TensorRT liga por padrão muda levemente o resultado em FP32.</li>
<li><b>Cooperative matrix no Vulkan</b> (<code>VK_KHR_cooperative_matrix</code> e <code>VK_NV_cooperative_matrix2</code>) deixa o ncnn usar esses tensor cores sem CUDA. Sem ela, o ncnn FP16 vai de ~9,8 para ~13 ms.</li>
<li><b>CPU híbrida:</b> se o runtime espalha threads nos E-cores, os P-cores ficam esperando os mais lentos a cada camada. Por isso as medições de CPU usam uma thread por P-core (<code>taskset -c 0,2,4,6,8,10</code>).</li>
<li><b>FP16 não acelera esta CPU:</b> a F16C só <i>converte</i> entre FP16 e FP32; a conta continua em FP32. Já a <b>AVX-VNNI</b> faz produto escalar de 8 bits no hardware, e é por isso que o INT8 compensa.</li>
<li><b>Teto teórico em FP32 dos 6 P-cores:</b> 6 núcleos × 32 FLOP/ciclo (duas unidades FMA de 256 bits) × frequência ≈ 800 GFLOPS a ~4,2 GHz. A YOLOv10s faz 21,7 GFLOPs por inferência em 640×640.</li>
<li><b>O notebook esquenta e reduz o clock:</b> numa sequência de medições sem pausa, a frequência dos P-cores caiu de 4,2 para 3,1 GHz e prejudicou quem rodou por último. Por isso as rodadas 3 e 4 usam resfriamento e ordem trocada entre rodadas.</li>
</ul>
""")

    # 3. Método comum
    S.append("""
<h2>3. Método comum</h2>
<h3>O modelo e o grafo</h3>
<p>Todos os backends partem do mesmo grafo: a YOLOv10s em modo eval, com as camadas fundidas (convolução + batch norm), entrada estática de <b>1×3×640×640</b>, batch 1, e saída <code>[1, 300, 6]</code> com <code>(x1, y1, x2, y2, score, classe)</code>. Um detalhe só apareceu na prática: nesta versão do Ultralytics, o checkpoint carrega com <code>end2end=False</code> e a saída vinha no formato antigo, <code>[1, 84, 8400]</code>, que exigiria NMS. O exportador liga o head one-to-one antes da fusão e confere o formato.</p>
<h3>Como a equivalência é validada</h3>
<ul>
<li><b>Referência fixa:</b> 8 imagens (<code>bus.jpg</code> e <code>zidane.jpg</code> do Ultralytics e 6 quadros da webcam), pré-processadas uma vez em Python (letterbox 640, preenchimento 114, RGB, /255) e executadas no PyTorch eager com TF32 desligado. Todos os backends leem os mesmos arquivos.</li>
<li><b>Tolerâncias</b>, contra o eager da mesma precisão: FP32 com caixas até 0,5 px e scores até 0,001; FP16 com caixas até 2 px e scores até 0,005; detecções com score ≥ 0,25 com mesma quantidade, mesma classe e IoU ≥ 0,95. As diferenças são medidas entre detecções pareadas por IoU, não pela posição da linha, porque o TopK pode ordenar de outro jeito linhas com scores quase iguais.</li>
<li><b>INT8</b> não atinge tolerâncias estritas, então foi julgado por concordância de detecções (IoU ≥ 0,80 e diferença de score ≤ 0,05) e pela perda de mAP50-95 no COCO (no máximo 1,0 ponto).</li>
<li><b>Pré-processamento em C++</b> é conferido à parte: os runners fazem o próprio letterbox a partir do JPEG e precisam reproduzir as detecções do eager (IoU ≥ 0,90).</li>
<li>As tolerâncias foram fixadas antes de cada rodada e nunca afrouxadas; falhas são reportadas com os números.</li>
</ul>
<h3>Protocolo de benchmark</h3>
<ul>
<li>Só a inferência, com a entrada já na memória do dispositivo. GPU: 50 iterações de aquecimento e 500 medidas; CPU: 20 e 200. Cada iteração é sincronizada. FPS = 1000 / latência média.</li>
<li>Cada variante roda no próprio processo. Configuração igual em todos os lados (cuDNN TF32 ligado, matmul TF32 desligado e autotune desligado, que são os padrões do PyTorch), impressa em cada resultado.</li>
<li>Nas rodadas 3 e 4: 60 s de resfriamento antes de cada medição, 3 rodadas com a ordem trocada e a mediana como resultado. Um monitor descarta a medição se outro processo estiver usando a CPU.</li>
</ul>
<h3>Como o trabalho foi organizado</h3>
<p>O design de cada rodada foi feito em conjunto e aprovado antes de qualquer implementação. O planejamento, as specs, a revisão e as medições finais ficaram com o orquestrador (Claude); a implementação foi delegada a um agente executor (GPT-6.1 Sol), e os diffs grandes passaram por um revisor independente (GPT-6 Astra). Todo número deste relatório foi medido de novo pelo orquestrador. Dois dos achados mais importantes, o TF32 assimétrico da rodada 1 e a causa da falha do TensorRT FP32, não foram apontados pelo revisor e saíram dessa conferência; a deriva térmica da rodada 3 foi medida pelo orquestrador e também apontada como risco pelo revisor.</p>
""")

    # 4. Webcam
    S.append("""
<h2>4. Ponto de partida: a webcam</h2>
<p>Antes de compilar qualquer coisa, a YOLOv10s rodou com o <code>predict</code> do Ultralytics na webcam do notebook, numa janela com FPS e memória de GPU. O loop ficou em <b>15 FPS</b>, mas a câmera sozinha também entregava 15 FPS: com pouca luz, o recurso <code>exposure_dynamic_framerate</code> dobra o tempo de exposição. Com ele desligado, o loop chegou a <b>30 FPS</b> (o máximo da câmera), com o modelo levando cerca de 13 ms por quadro e 232 MiB de memória de GPU. Como a câmera limita o loop muito antes do modelo, o experimento passou a medir <b>só a inferência</b>.</p>
<p class="small">O <code>predict</code> levava cerca de 7 ms porque faz letterbox retangular: o quadro 1280×720 vira 640×384. Nas rodadas a entrada é sempre 640×640, cerca de 1,7× mais pixels, por isso o eager FP32 fica perto de 10 ms.</p>
""")

    # 5. Rodada 1
    S.append("""
<h2 class="newpage">5. Rodada 1: AOTInductor na GPU</h2>
<p>O AOTInductor é o compilador ahead-of-time do PyTorch 2. O fluxo é <code>torch.export</code> → <code>aoti_compile_and_package</code>, que gera um pacote <code>.pt2</code> com kernels Triton compilados, um wrapper C++ e os pesos embutidos. Um runner C++ (<code>yolo_aoti</code>) carrega o pacote com <code>AOTIModelPackageLoader</code> e executa sem Python, mas ainda com a libtorch.</p>
<h3>Problemas resolvidos</h3>
<ul>
<li>O CUDA do sistema (13.4) é diferente do CUDA do PyTorch (12.8), e o AOTInductor escolhia sozinho os headers do <code>/opt/cuda</code>. Solução: um <code>CUDA_HOME</code> local com os headers e o runtime das wheels 12.8; o header <code>nv/target</code> veio da wheel <code>nvidia-cuda-cccl-cu12</code>. O build final não toca no <code>/opt/cuda</code>.</li>
<li>O CMake da libtorch tenta habilitar CUDA e procurar o nvcc. O runner só tem <code>.cpp</code>, então as bibliotecas são importadas direto da wheel.</li>
<li>O GCC 16, mais novo que o suportado oficialmente, compilou sem patch nem flags especiais.</li>
<li><b>Comparação justa:</b> a primeira medição deu 1,29× em FP32, mas o eager estava com TF32 desligado e o C++ com o padrão da libtorch, que liga TF32 nas convoluções. Com a mesma configuração dos dois lados, o eager FP32 cai de 10,9 para 9,9 ms e o ganho real fica em 1,18×.</li>
</ul>
<p><b>Resultado:</b> 16/16 validações PASS (maior diferença em FP32: 0,035 px nas caixas e 0,0006 no score). Pacotes de 29,8 MiB (FP32) e 16,0 MiB (FP16), compilados em 10 a 13 s com cache. FP32 a 8,46 ms (1,18×) e FP16 a 3,92 ms (1,43×). Ponto fraco: a primeira inferência leva cerca de 2 s.</p>
""")

    # 6. Rodada 2
    S.append("""
<h2>6. Rodada 2: TensorRT na GPU</h2>
<p>Caminho: <code>torch.onnx.export</code> (opset 18) → ONNX → <code>trtexec --stronglyTyped</code> → engine <code>.plan</code>. O runner <code>yolo_trt</code> desserializa o engine e chama <code>enqueueV3</code> num stream CUDA. Ele depende só de <code>libnvinfer</code> e do runtime CUDA 13.4, <b>sem libtorch e sem Python</b>. O TensorRT 11.4 veio do AUR. No TensorRT 11 o FP16 por flag do builder ficou obsoleto, então o FP16 vem do próprio ONNX exportado em half.</p>
<h3>O caso do TF32</h3>
<p>O engine FP32 padrão falhou numa única imagem (<code>webcam_04</code>, uma única detecção): a diferença de score foi 0,0022 no primeiro build e 0,0019 num rebuild, contra o limite de 0,001, com a mesma classe e caixas praticamente iguais (IoU 0,9998). O ONNX Runtime, rodando o mesmo ONNX, bate com o eager a menos de 2e-7, e o <code>trtexec</code> reproduz o erro sozinho, então a diferença não vem da exportação nem do runner. A causa é que o TensorRT <b>permite TF32 por padrão</b> em tensores FP32, e o engine contém kernels de convolução e GEMM em TF32, enquanto a referência é FP32 estrito. Um engine do mesmo ONNX com <code>--noTF32</code> passa nas 8 imagens com 3e-5 px de diferença, praticamente bit a bit, por cerca de 0,5 ms a mais. O FAIL do engine padrão foi mantido: é uma questão de tolerância de score, não um erro de detecção nesta amostra.</p>
<p><b>Resultado:</b> FP32 a 7,39 ms (1,36×; 7,86 ms sem TF32) e <b>FP16 a 2,94 ms (340 FPS, 1,90×)</b>. Menor RSS entre eager, AOTInductor e TensorRT (~650 MiB). O custo é o build de 55 a 60 s por engine e um <code>.plan</code> de 80 a 110 MiB que vale só para esta GPU e esta versão do TensorRT.</p>
""")

    # 7. Rodada 3
    S.append("""
<h2 class="newpage">7. Rodada 3: CPU</h2>
<p>Objetivo: medir o desempenho de referência em CPU com runtimes prontos, comparado com o eager e com o teto teórico. Esse é o alvo para um futuro engine próprio em C++. FP16 ficou de fora porque não acelera esta CPU; o tipo reduzido testado foi <b>INT8 com AVX-VNNI</b>, quantizado com o NNCF (OpenVINO). O benchmark foi em Python, porque com dezenas de ms por inferência o custo da chamada é desprezível (menos de 0,1%).</p>
<h3>INT8 e acurácia</h3>
<p>A calibração usou 300 imagens do COCO val2017 (posições 501 a 800 por id), disjuntas das 500 de avaliação. O preset <code>performance</code> do NNCF falhou no critério de detecção; excluir o pós-processamento da quantização piorou o mAP; o preset <code>mixed</code> passou nos dois critérios. Modelo de 28,0 MiB (FP32) para 7,7 MiB (INT8).</p>
""")
    S.append(table(["Variante (500 imagens do COCO val2017)", "mAP50-95", "mAP50"], [
        ["PyTorch eager FP32", "49,33", "65,26"],
        ["OpenVINO FP32", "49,33", "65,26"],
        ["OpenVINO INT8", "49,02 (−0,31)", "65,31"],
    ], num_cols=(1, 2)))
    S.append("""
<h3>Controle térmico</h3>
<p>A primeira tabela foi medida com as 5 variantes em sequência, e a frequência dos P-cores caiu de 4,21 para 3,09 GHz ao longo da fila. Rodando cada variante depois de um resfriamento, o INT8 foi de 19,35 ms (a 3,36 GHz) para 15,17 ms (a 4,19 GHz). O script final faz 60 s de resfriamento antes de cada medição, 3 rodadas com ordem trocada e usa a mediana; as 15 medições foram aceitas, com variação de no máximo 3%. A suposta lentidão do OpenVINO no modo padrão (27%) era só efeito de ter rodado por último.</p>
""")
    S.append(table(["Variante", "Média", "p95", "FPS", "Ganho", "% do pico FP32", "RAM (MiB)", "Modelo"], [
        ["PyTorch eager FP32", "55,56 ms", "64,51", "18,0", "1,00×", "47%", "929", "15,9 MiB"],
        ["ncnn FP32 (rodada 4)", "56,89 ms", "60,57", "17,6", "0,98×", "—", "223", "27,8 MiB"],
        ["ONNX Runtime FP32", "43,63 ms", "49,96", "22,9", "1,27×", "63%", "219", "28,4 MiB"],
        ["OpenVINO FP32 (6 P-cores)", "40,20 ms", "45,73", "24,9", "1,38×", "70%", "215", "28,0 MiB"],
        ["OpenVINO FP32 padrão", "39,84 ms", "41,12", "25,1", "1,39×", "70%", "214", "28,0 MiB"],
        ["<b>OpenVINO INT8</b>", "<b>14,91 ms</b>", "16,04", "<b>67,1</b>", "<b>3,73×</b>", "—", "172", "7,7 MiB"],
    ], num_cols=(1, 2, 3, 4, 5, 6), best=()))
    S.append(f'<div class="figure">{cpu_chart}</div>')
    S.append("""<p class="small">Medianas de 3 rodadas com controle térmico, 6 threads fixadas nos P-cores. % do pico = 21,7 GFLOPs / latência, dividido por 6 × 32 FLOP/ciclo × frequência medida. O INT8 equivale a 177% do pico FP32 em vazão, o que não é utilização do hardware FP32.</p>""")

    # 8. Rodada 4
    S.append("""
<h2>8. Rodada 4: ncnn e Vulkan</h2>
<p>O Vulkan é a API de GPU independente de fabricante: o mesmo binário roda em NVIDIA, AMD, Intel e celular, sem CUDA. O modelo foi convertido com o <code>pnnx</code> para o formato do ncnn (<code>.param</code> + <code>.bin</code>, 27,8 MiB). O ncnn não suporta o TopK e o Gather do head do YOLOv10, então o grafo termina no head bruto e a seleção dos 300 melhores roda em C++, <b>batendo bit a bit</b> com o Python. Esse pós-processamento custa de 0,9 a 2,5 ms e está somado aos resultados. O runner <code>yolo_ncnn</code> depende só do ncnn: sem Torch, sem Python e sem CUDA.</p>
<p>A <b>cooperative matrix</b> foi confirmada em uso na RTX 3050: os shaders de convolução usam <code>OpCooperativeMatrixMulAddKHR</code>, e desligá-la leva o FP16 de 9,8 para 13,0 ms (medições únicas).</p>
""")
    S.append(table(["Variante ncnn (medianas)", "Entrada na GPU", "Entrada na CPU (com cópias)", "GPU (MiB)", "Validação"], [
        ["<b>RTX 3050 FP16</b> (cooperative matrix)", "<b>9,69 ms (103 FPS)</b>", "10,17 ms", "~124", "8/8"],
        ["RTX 3050 FP32", "18,55 ms (54 FPS)", "19,19 ms", "~196", "8/8"],
        ["CPU FP32 (6 P-cores)", "—", "56,89 ms", "—", "8/8"],
        ["Intel integrada FP16 <span class='small'>(uma rodada)</span>", "229,7 ms", "230,5 ms", "—", "6/8"],
        ["Intel integrada FP32 <span class='small'>(uma rodada)</span>", "433,0 ms", "437,0 ms", "—", "8/8"],
    ], num_cols=(1, 2, 3)))
    S.append("""
<div class="note"><b>GPU integrada:</b> o benchmark controlado da Intel foi <b>interrompido por decisão do Nicolas, por ser muito lenta</b>: cada medição levava de 2 a 4 minutos. Os números acima são de uma única rodada, sem mediana. Em FP16 ela é cerca de 15× mais lenta que a CPU com OpenVINO INT8. Duas imagens falharam em FP16 por diferença de score (0,0051 e 0,0093 contra 0,005); com FP16 só no armazenamento e a conta em FP32, passaram 8/8.</div>
""")

    # 9. Comparação geral
    S.append("<h2 class='newpage'>9. Comparação geral</h2>")
    S.append(f'<div class="figure">{gpu_chart}</div>')
    S.append(table(["Dispositivo", "Backend", "Precisão", "Latência", "FPS", "Dependências em execução", "Artefato"], [
        ["RTX 3050", "PyTorch eager", "FP32 / FP16", "10,02 / 5,59 ms", "100 / 179", "Python, PyTorch, CUDA 12.8", ".pt 15,9 MiB"],
        ["RTX 3050", "AOTInductor", "FP32 / FP16", "8,46 / 3,92 ms", "118 / 255", "libtorch, CUDA 12.8; sem Python", ".pt2 29,8 / 16,0 MiB"],
        ["RTX 3050", "<b>TensorRT</b>", "FP32 / FP16", "<b>7,39 / 2,94 ms</b>", "<b>135 / 340</b>", "libnvinfer, CUDA 13.4", ".plan 108,5 / 81,6 MiB"],
        ["RTX 3050", "ncnn Vulkan", "FP32 / FP16", "18,55 / 9,69 ms", "54 / 103", "libncnn, loader Vulkan", "27,8 MiB"],
        ["CPU", "ONNX Runtime", "FP32", "43,63 ms", "22,9", "libonnxruntime", ".onnx 28,4 MiB"],
        ["CPU", "OpenVINO", "FP32 / INT8", "40,20 / <b>14,91 ms</b>", "24,9 / <b>67,1</b>", "runtime OpenVINO", "IR 28,0 / 7,7 MiB"],
        ["CPU", "ncnn", "FP32", "56,89 ms", "17,6", "libncnn", "27,8 MiB"],
        ["GPU integrada", "ncnn Vulkan", "FP32 / FP16", "~433 / ~230 ms", "2,3 / 4,4", "libncnn, loader Vulkan", "27,8 MiB"],
    ], num_cols=(3, 4)))
    S.append("""<p class="small">As linhas de eager, AOTInductor e TensorRT vêm de uma mesma sessão sem pausas (variação esperada de cerca de 3%); as de CPU e ncnn são medianas com controle térmico. Comparações entre sessões são aproximadas.</p>""")
    S.append("<h3>Validação consolidada</h3>")
    S.append(table(["Artefato", "PASS", "Observação"], [
        ["AOTInductor FP32 / FP16", "8/8 · 8/8", "maior diferença FP32: 0,035 px, score 0,0006"],
        ["TensorRT FP32 padrão", "<span class='warn'>7/8</span>", "TF32: score 0,0019–0,0022 numa imagem (limite 0,001)"],
        ["TensorRT FP32 sem TF32 / FP16", "8/8 · 8/8", "sem TF32: 3e-5 px, praticamente bit a bit"],
        ["CPU: eager, ONNX Runtime, OpenVINO FP32 (2 configurações)", "32/32", "tolerâncias estritas"],
        ["OpenVINO INT8", "8/8", "critério INT8; mAP −0,31 ponto"],
        ["ncnn RTX 3050 FP32 / FP16", "8/8 · 8/8", "pós-processamento em C++ bit a bit"],
        ["ncnn Intel FP32 / FP16", "8/8 · <span class='warn'>6/8</span>", "FP16: score 0,0051 e 0,0093 (limite 0,005)"],
        ["ncnn CPU FP32", "8/8", ""],
        ["<b>Total</b>", "<b>117/120</b>", "o <code>detect</code> com pré-processamento em C++ passou em todos os artefatos"],
    ]))

    # 10. Problemas e lições
    S.append("""
<h2>10. Problemas e lições</h2>
<ul>
<li><b>Configuração assimétrica distorce comparações.</b> O TF32 ligado de um lado e desligado do outro inflou o ganho do AOTInductor de 1,18× para 1,29×. Toda medição passou a imprimir a configuração usada.</li>
<li><b>A temperatura do notebook é uma variável.</b> Medições em sequência penalizaram quem rodou por último em até 25%. Resfriamento, ordem trocada e mediana resolveram.</li>
<li><b>O ambiente é ruidoso.</b> Navegador, builds e testes de outros projetos invalidam medições de CPU; o monitor de carga foi essencial.</li>
<li><b>Dois CUDAs convivem</b> se ficarem em processos separados e se o build do AOTInductor apontar para os headers das wheels.</li>
<li><b>Instalar o TensorRT pelo AUR</b> exigiu atualizar o sistema (mirror desatualizado), reiniciar (driver novo) e compilar com <code>-j1</code> (corrida entre as wheels Python no build paralelo).</li>
<li><b>Exportação não é trivial:</b> o head end2end precisa ser ligado antes da fusão; o ncnn não tem TopK; o TensorRT 11 só aceita FP16 vindo do próprio ONNX.</li>
</ul>
""")

    # 11. Conclusão
    S.append("""
<h2>11. Conclusão</h2>
<h3>Respostas às perguntas</h3>
<ul>
<li><b>Viabilidade: sim.</b> A YOLOv10s exportou e compilou em todos os caminhos sem operações customizadas (no ncnn, com o TopK em C++). As saídas ficaram dentro das tolerâncias, com duas exceções de causa conhecida: o TF32 do TensorRT e o FP16 na GPU integrada.</li>
<li><b>Dependências: dependem do backend.</b> O AOTInductor elimina o Python, mas não a libtorch. O TensorRT deixa só <code>libnvinfer</code> e o runtime CUDA. O ncnn deixa só a própria biblioteca e o loader Vulkan, e roda em qualquer GPU. Em todos os casos os pesos ficam embutidos no artefato.</li>
<li><b>Ganho: real e maior em precisão reduzida.</b> Na GPU NVIDIA, o TensorRT FP16 dá 1,90× sobre o eager. Na CPU, o OpenVINO INT8 dá 3,73× sobre o eager e 2,70× sobre o OpenVINO FP32, perdendo só 0,31 ponto de mAP. Já o Vulkan na NVIDIA ficou abaixo do eager (0,58× em FP16).</li>
</ul>
<h3>O que usar em cada alvo</h3>
<ul>
<li><b>GPU NVIDIA:</b> TensorRT, se for aceitável gerar o engine por GPU e por versão. AOTInductor, se o projeto já está no PyTorch e quer build rápido.</li>
<li><b>CPU x86:</b> OpenVINO INT8, que dá 67 FPS nesta CPU, mais que os 30 FPS da webcam.</li>
<li><b>GPU de qualquer fabricante:</b> ncnn com Vulkan funciona e é leve, mas paga caro na NVIDIA; vale quando CUDA não é opção.</li>
<li><b>GPU integrada desta máquina:</b> não vale a pena para este modelo; a CPU é muito mais rápida.</li>
</ul>
<p>Para a integração com a Enty (PBI-096), o resultado sugere um backend em torno de um formato intermediário (ONNX) e de um runtime nativo trocável por alvo, já que o melhor caminho muda conforme o hardware.</p>
<h3>Limitações</h3>
<ul>
<li>Uma única máquina, sem clock travado. Só batch 1 e entrada estática 640×640.</li>
<li>As 8 imagens de validação conferem equivalência numérica, não acurácia; a acurácia (mAP) só foi medida na comparação INT8 da CPU.</li>
<li>Os artefatos do TensorRT e do AOTInductor são específicos da arquitetura da GPU (e, no TensorRT, da versão).</li>
</ul>
<h3>Próximo passo: um engine próprio em C++</h3>
<p>Escrever, a partir do ONNX, um engine de inferência em C++ para a CPU: primeiro uma versão de referência validada com o mesmo código comum, depois um micro-kernel de GEMM com AVX2, fusão convolução + SiLU, layout blocado e threads, e por fim INT8 com AVX-VNNI. As metas são as medidas aqui: <b>abaixo de ~40 ms em FP32</b>, onde o OpenVINO já usa ~70% do pico e sobra pouco espaço, e <b>perto de 14,9 ms em INT8</b>, onde está a alavanca maior. O valor principal desse engine é um artefato sem nenhuma dependência e o domínio do caminho todo, que é exatamente o que a Enty precisa.</p>
""")

    # Apêndice
    S.append("<h2>12. Apêndice: versões e reprodução</h2>")
    S.append(table(["Componente", "Versão"], [
        ["PyTorch / Ultralytics", "2.11.0+cu128 / 8.4.210"],
        ["TensorRT / cuDNN (sistema)", "11.4.0.106 / 9.28.1"],
        ["ONNX / ONNX Runtime / onnxscript", "1.23.2 / 1.31.1 / 0.7.2"],
        ["OpenVINO / NNCF / pycocotools", "2026.4.1 / 3.4.0 / 2.0.11"],
        ["ncnn / pnnx", "20261020 / 20261020"],
        ["Driver NVIDIA / Mesa", "620.70 / 26.2.6"],
        ["CUDA", "13.4 (sistema, TensorRT) e 12.8 (wheels, PyTorch e AOTInductor)"],
        ["Compiladores", "GCC 16.2.1, CMake 4.4, Ninja 1.13, Python 3.12"],
    ]))
    S.append("""<p>O passo a passo completo para reproduzir cada rodada (pacotes, comandos, saídas esperadas e solução de problemas) está no <code>README.md</code> e em <code>docs/REPRODUCE.md</code>, em inglês. Os relatórios técnicos de cada rodada estão em <code>docs/aoti-results.md</code>, <code>docs/tensorrt-results.md</code>, <code>docs/cpu-results.md</code> e <code>docs/ncnn-vulkan-results.md</code>. Este relatório é gerado por <code>docs/report/build_report.py</code>.</p>""")

    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLOv10s compilada para runtimes nativos</title>
<style>{CSS}</style></head>
<body>{"".join(S)}</body></html>
"""


def main():
    html = build_html()
    HTML_OUT.write_text(html, encoding="utf-8")
    import weasyprint
    weasyprint.HTML(string=html, base_url=str(HERE)).write_pdf(PDF_OUT)
    print(f"HTML: {HTML_OUT}\nPDF:  {PDF_OUT}")


if __name__ == "__main__":
    main()
