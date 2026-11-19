"""Draw the DRMNet fusion architecture as a publication-ready SVG.

Run from any directory: python figures/create_architecture_figure.py
The SVG is intentionally self-contained, editable, and independent of fonts or
third-party Python packages. Scientific labels follow the fusion variant code.
"""

from html import escape
from pathlib import Path


OUT = Path(__file__).with_name("drmnet_fusion_architecture.svg")
W, H = 3900, 2300
C = {
    "ink": "#263541", "muted": "#526475", "line": "#637582",
    "hair": "#B8C6CF", "blue": "#E7F0F6", "blue_edge": "#6F9AB7",
    "teal": "#DDF2EE", "teal_edge": "#168A83", "teal_deep": "#0F7772",
    "orange": "#FCEAD9", "orange_edge": "#CA772F", "orange_deep": "#B96D27",
    "purple": "#EEE9F6", "purple_edge": "#9173AD", "gray": "#F4F6F7",
    "white": "#FFFFFF", "ct": "#D7DDE0",
}
parts = []


def add(s):
    parts.append(s)


def rect(x, y, w, h, fill="white", stroke="hair", r=14, sw=2, dash=None):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
        f'fill="{C.get(fill, fill)}" stroke="{C.get(stroke, stroke)}" '
        f'stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else "") + '/>')


def line(x1, y1, x2, y2, color="line", sw=2.5, dash=None, arrow=False):
    add(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" '
        f'stroke="{C.get(color, color)}" stroke-width="{sw}" '
        f'stroke-linecap="round" stroke-linejoin="round"'
        + (f' stroke-dasharray="{dash}"' if dash else "")
        + (' marker-end="url(#arrow)"' if arrow else "") + '/>')


def path(d, color="line", sw=2.5, dash=None, arrow=False):
    add(f'<path d="{d}" fill="none" stroke="{C.get(color, color)}" '
        f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"'
        + (f' stroke-dasharray="{dash}"' if dash else "")
        + (' marker-end="url(#arrow)"' if arrow else "") + '/>')


def text(x, y, s, size=28, weight=400, color="ink", anchor="start", italic=False):
    add(f'<text x="{x}" y="{y}" fill="{C.get(color, color)}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'dominant-baseline="middle"' + (' font-style="italic"' if italic else '')
        + f'>{escape(s)}</text>')


def lines(x, y, items, size=25, step=34, weight=400, color="ink", anchor="middle"):
    for i, s in enumerate(items):
        text(x, y + i * step, s, size, weight, color, anchor)


def module(x, y, w, h, title, sub=(), fill="blue", edge="blue_edge", title_size=29, sub_size=24):
    rect(x, y, w, h, fill, edge, 16, 2.4)
    if sub:
        title_y = y + h / 2 - (len(sub) * 27 + 10) / 2
        text(x + w / 2, title_y, title, title_size, 700, anchor="middle")
        lines(x + w / 2, title_y + 33, sub, sub_size, 27)
    else:
        text(x + w / 2, y + h / 2, title, title_size, 700, anchor="middle")


def dot(x, y, color="line", radius=5):
    add(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{C.get(color, color)}"/>')


def node(x, y, label, edge="line", radius=23, size=27):
    add(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="white" '
        f'stroke="{C.get(edge, edge)}" stroke-width="2.4"/>')
    text(x, y + 1, label, size, 700, edge, "middle")


def tag(x, y, label, color):
    rect(x, y, 58, 42, "white", color, 12, 2)
    text(x + 29, y + 22, label, 24, 700, color, "middle")


def feature_stack(x, y, w=86, h=63, color="blue_edge"):
    for offset in (14, 7, 0):
        rect(x + offset, y - offset, w, h, "white", color, 5, 2)
    for i in range(3):
        line(x + 16, y + 14 + i * 14, x + w - 14, y + 14 + i * 14, "hair", 1.4)


def ct_slice(x, y, w, h, boxes=False):
    rect(x, y, w, h, "#D5DADC", "#7B8991", 7, 1.8)
    add(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w*.45}" ry="{h*.46}" fill="#A7B0B4"/>')
    add(f'<path d="M{x+w*.43} {y+h*.16} C{x+w*.09} {y+h*.08}, {x+w*.06} {y+h*.78}, {x+w*.34} {y+h*.84} C{x+w*.50} {y+h*.72}, {x+w*.47} {y+h*.35}, {x+w*.43} {y+h*.16} Z" fill="#47545B"/>')
    add(f'<path d="M{x+w*.57} {y+h*.16} C{x+w*.91} {y+h*.08}, {x+w*.94} {y+h*.78}, {x+w*.66} {y+h*.84} C{x+w*.50} {y+h*.72}, {x+w*.53} {y+h*.35}, {x+w*.57} {y+h*.16} Z" fill="#47545B"/>')
    add(f'<ellipse cx="{x+w*.5}" cy="{y+h*.52}" rx="{w*.055}" ry="{h*.25}" fill="#B8C2C7"/>')
    add(f'<circle cx="{x+w*.28}" cy="{y+h*.44}" r="{w*.032}" fill="#B8C2C7"/>')
    add(f'<circle cx="{x+w*.73}" cy="{y+h*.63}" r="{w*.025}" fill="#BCC6C9"/>')
    if boxes:
        rect(x + w*.205, y + h*.36, w*.18, h*.19, "none", "orange_edge", 1, 3)
        rect(x + w*.645, y + h*.56, w*.17, h*.16, "none", "teal_edge", 1, 3)


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="fig-title fig-desc">')
add('<title id="fig-title">Multi-directional state-space modeling and gated frequency-aware fusion for 3D lung nodule detection</title>')
add('<desc id="fig-desc">Three panels show the DRMNet-based detector, its multi-directional Mamba context module, and its gated spatially adaptive 3D frequency-aware fusion module.</desc>')
add('<defs><marker id="arrow" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M1 1 L9 5.5 L1 10" fill="none" stroke="#526475" stroke-width="1.7"/></marker></defs>')
rect(0, 0, W, H, "white", "white", 0, 0)
text(70, 62, "Multi-directional State-Space Modeling and Gated Frequency-aware Fusion", 43, 700)
text(70, 111, "for 3D Lung Nodule Detection", 39, 700)

# PANEL A --------------------------------------------------------------------
rect(55, 150, 3790, 970, "white", "hair", 22, 2)
tag(84, 177, "(a)", "blue_edge")
text(158, 199, "Overall detection framework", 35, 700)
text(3820, 199, "Displayed sizes correspond to one 1 × 128 × 128 × 128 training patch", 24, 400, "muted", "end")

# Long skips are routed above the encoder/decoder blocks.
path("M420 368 V231 H3160 V487 H3150", "blue_edge", 3, arrow=True)
text(1705, 215, "E1 stride-2 feature → ROI", 23, 600, "blue_edge", "middle")
path("M645 368 V267 H2030 V356", "blue_edge", 3)
text(1484, 253, "E2 skip, 32³", 22, 600, "blue_edge", "middle")
path("M870 368 V309 H1549 V670 H1363 V639", "blue_edge", 3, arrow=True)
text(1287, 294, "E3 skip, 16³", 22, 600, "blue_edge", "middle")

ct_slice(102, 388, 150, 178)
text(177, 591, "CT volume", 26, 700, anchor="middle")
text(177, 624, "1 × 128 × 128 × 128", 22, 400, "muted", "middle")
line(252, 475, 316, 475, arrow=True)

for x, stage, size, context, fill, edge in [
    (330, "E1", "64³ · stride 2", "Single-direction", "blue", "blue_edge"),
    (555, "E2", "32³ · stride 4", "Single-direction", "blue", "blue_edge"),
    (780, "E3", "16³ · stride 8", "Multi-directional", "teal", "teal_edge"),
    (1005, "E4", "8³ · stride 16", "Multi-directional", "teal", "teal_edge"),
]:
    rect(x, 368, 180, 213, fill, edge, 15, 2.4)
    feature_stack(x + 47, 400, 75, 55, edge)
    text(x + 90, 484, stage, 30, 700, anchor="middle")
    text(x + 90, 519, size, 22, 600, "muted", "middle")
    text(x + 90, 542, context, 18.5, 600, edge, "middle")
    text(x + 90, 567, "Mamba", 18.5, 600, edge, "middle")
for a, b in [(510, 555), (735, 780), (960, 1005), (1185, 1235)]:
    line(a, 475, b - 8, 475, arrow=True)

# Decoder level 1.
rect(1235, 356, 330, 334, "blue", "blue_edge", 17, 2.4)
text(1400, 389, "Decoder level 1 · 16³", 27, 700, anchor="middle")
module(1260, 414, 280, 69, "Upsample ×2", fill="white", edge="blue_edge", title_size=24)
module(1260, 491, 280, 69, "Existing Mamba", fill="white", edge="blue_edge", title_size=24)
node(1363, 615, "C", "blue_edge", 21, 24)
module(1402, 580, 138, 70, "Conv refine", fill="white", edge="blue_edge", title_size=21)
line(1400, 483, 1400, 489, arrow=True)
line(1400, 560, 1400, 589, arrow=True)
line(1384, 615, 1400, 615, arrow=True)
path("M1235 475 V448 H1257", "line", 2.5, arrow=True)
path("M1549 356 V670 H1363 V639", "blue_edge", 3, arrow=True)

# Decoder level 2: frequency refinement only here, before concatenate.
rect(1615, 356, 430, 334, "blue", "blue_edge", 17, 2.4)
text(1830, 389, "Decoder level 2 · 32³", 27, 700, anchor="middle")
module(1640, 410, 380, 65, "Upsample ×2 + existing Mamba", fill="white", edge="blue_edge", title_size=23)
module(1640, 490, 380, 88, "Gated 3D frequency-aware", ("fusion of E2 skip and decoder",), "orange", "orange_edge", 24, 20)
node(1830, 626, "C", "blue_edge", 21, 24)
module(1875, 591, 145, 70, "Conv refine", fill="white", edge="blue_edge", title_size=21)
line(1830, 475, 1830, 487, arrow=True)
path("M1740 578 V626 H1807", "orange_edge", 2.5, arrow=True)
path("M1850 578 V600 H1830 V602", "orange_edge", 2.5, arrow=True)
line(1851, 626, 1872, 626, arrow=True)
text(1830, 674, "H′ and L′ → concatenate", 20, 400, "muted", "middle")
path("M2030 356 V530 H2023", "blue_edge", 3, arrow=True)
path("M1540 615 H1585 V442 H1637", "line", 2.7, arrow=True)

# Feature, RPN, proposals, ROI, RCNN, final NMS.
module(2090, 420, 165, 174, "Detection", ("feature", "128 channels", "stride 4"), "blue", "blue_edge", 26, 22)
path("M2020 626 H2065 V500 H2086", "line", 2.7, arrow=True)
module(2310, 412, 190, 183, "RPN", ("Objectness", "+ 3D box offsets"), "purple", "purple_edge", 29, 23)
line(2255, 500, 2306, 500, arrow=True)
module(2555, 422, 200, 163, "Candidates", ("Decode boxes", "+ NMS"), "purple", "purple_edge", 26, 23)
line(2500, 500, 2551, 500, arrow=True)

rect(2810, 355, 365, 430, "white", "purple_edge", 18, 2.5)
text(2992, 388, "Proposal-guided ROI", 27, 700, anchor="middle")
text(2992, 421, "feature extraction", 27, 700, anchor="middle")
module(2838, 458, 309, 58, "Crop at strides 4 and 2", fill="gray", edge="hair", title_size=22)
module(2838, 536, 309, 58, "Upsample stride-4 crop ×2", fill="gray", edge="hair", title_size=21)
module(2838, 614, 309, 58, "Fuse with E1 crop", fill="gray", edge="hair", title_size=22)
module(2838, 692, 309, 58, "Adaptive max pool · 7 × 7 × 7", fill="purple", edge="purple_edge", title_size=20.5)
path("M3160 355 V487 H3150", "blue_edge", 3, arrow=True)
for y in (516, 594, 672):
    line(2992, y, 2992, y + 17, arrow=True)
text(3020, 771, "64 channels per ROI", 21, 400, "muted", "end")
path("M2755 500 H2785 V470 H2835", "purple_edge", 2.5, arrow=True)
text(2780, 444, "proposals", 19, 600, "muted", "middle")
path("M2255 594 V755 H2790 V559 H2824 V487 H2835", "blue_edge", 3, arrow=True)
text(2485, 739, "stride-4 feature", 21, 600, "blue_edge", "middle")
text(3015, 337, "stride-2 E1 feature", 21, 600, "blue_edge", "middle")

rect(3230, 417, 230, 230, "purple", "purple_edge", 16, 2.4)
text(3345, 450, "RCNN", 29, 700, anchor="middle")
module(3250, 477, 190, 59, "Classification", fill="white", edge="purple_edge", title_size=22)
module(3250, 560, 190, 59, "Box regression", fill="white", edge="purple_edge", title_size=22)
line(3175, 560, 3226, 560, arrow=True)
module(3530, 492, 118, 113, "Final", ("NMS",), "purple", "purple_edge", 24, 23)
path("M3440 507 H3500 V520 H3527", "purple_edge", 2.5, arrow=True)
path("M3440 590 H3500 V577 H3527", "purple_edge", 2.5, arrow=True)
line(3648, 549, 3682, 549, arrow=True)
ct_slice(3685, 430, 135, 161, True)
text(3752, 619, "Detected", 23, 700, anchor="middle")
text(3752, 650, "nodules", 23, 700, anchor="middle")
text(3752, 682, "illustrative slice", 18, 400, "muted", "middle")

# Dashed, non-directional callouts; the data arrows above remain solid.
path("M1095 582 V1020 H905 V1160", "teal_edge", 2.5, "11 10")
path("M1830 690 V1070 H2910 V1160", "orange_edge", 2.5, "11 10")
dot(1095, 582, "teal_edge", 6)
dot(1830, 690, "orange_edge", 6)
text(970, 1001, "see (b)", 22, 600, "teal_deep")
text(2760, 1050, "see (c)", 22, 600, "orange_deep")

# Compact figure-wide legend.
text(100, 877, "Notation", 24, 700)
line(233, 877, 306, 877, arrow=True)
text(322, 877, "feature flow", 22)
node(545, 877, "C", "blue_edge", 19, 22)
text(577, 877, "concatenate", 22)
node(765, 877, "+", "line", 19, 24)
text(797, 877, "element-wise add", 22)
node(1080, 877, "×", "line", 19, 22)
text(1113, 877, "element-wise multiply", 22)
line(1425, 877, 1510, 877, "teal_edge", 2.5, "11 10")
text(1526, 877, "module detail leader", 22)
text(100, 947, "Complementary roles:", 24, 700)
text(367, 947, "deep volumetric context modeling", 24, 600, "teal_deep")
text(766, 947, "+", 24, 700)
text(800, 947, "spatially adaptive semantic-detail fusion", 24, 600, "orange_deep")

# PANEL B --------------------------------------------------------------------
rect(55, 1160, 1865, 1075, "white", "hair", 22, 2)
tag(84, 1185, "(b)", "teal_edge")
text(158, 1206, "Multi-directional Mamba", 35, 700)
text(100, 1290, "Input  B × C × D × H × W", 25, 700, "teal_deep")
module(105, 1450, 190, 110, "Input X", ("B × C × D × H × W",), "teal", "teal_edge", 25, 19)
dot(331, 1505, "teal_edge", 5)
line(295, 1505, 331, 1505, arrow=True)
path("M331 1505 V1395 H372", "teal_edge", 2.5, arrow=True)
path("M331 1505 V1903 H372", "teal_edge", 2.5, arrow=True)
text(390, 1355, "GLOBAL CONTEXT · six sequences", 23, 700, "teal_deep")
rect(372, 1385, 1480, 438, "#F6FBFA", "#B6DCD7", 16, 2)
for y, order in [(1445, "(D, H, W)"), (1550, "(H, W, D)"), (1655, "(W, D, H)")]:
    module(399, y - 36, 177, 73, order, fill="white", edge="teal_edge", title_size=22)
    for j, reverse in enumerate((False, True)):
        yy = y - 16 + j * 32
        for k in range(7):
            rect(610 + k * 21, yy - 10, 18, 20, "teal", "teal_edge", 2, 1)
        line(770 if reverse else 604, yy, 604 if reverse else 773, yy,
             "teal_deep", 2, arrow=True)
        text(800, yy, "reverse" if reverse else "forward", 19, 600, "teal_deep")
    line(576, y, 598, y, arrow=True)
path("M884 1445 H918 V1547 H948", "teal_edge", 2, arrow=True)
line(884, 1550, 948, 1550, "teal_edge", 2, arrow=True)
path("M884 1655 H918 V1553 H948", "teal_edge", 2, arrow=True)
module(950, 1483, 180, 134, "LayerNorm", ("shared Mamba", "6 passes"), "teal", "teal_edge", 23, 20)
line(1130, 1550, 1164, 1550, arrow=True)
module(1168, 1483, 180, 134, "Restore each", ("sequence to", "D, H, W"), "white", "teal_edge", 22, 19)
line(1348, 1550, 1375, 1550, arrow=True)
module(1380, 1483, 215, 134, "6 learned", ("softmax weights", "weighted sum"), "teal", "teal_edge", 22, 19)
line(1595, 1550, 1625, 1550, arrow=True)
module(1630, 1483, 177, 134, "LayerNorm", fill="white", edge="teal_edge", title_size=21)
text(1100, 1742, "Direction weights are learned parameters, not input-conditioned attention maps", 21, 400, "muted", "middle")
text(1100, 1781, "The six traversals share one Mamba within a block; E3 and E4 use independent blocks", 21, 400, "muted", "middle")

text(390, 1870, "LOCAL DETAIL", 23, 700, "teal_deep")
rect(372, 1890, 1195, 108, "#F6FBFA", "#B6DCD7", 16, 2)
module(405, 1905, 330, 77, "Depthwise 3 × 3 × 3 conv", fill="white", edge="teal_edge", title_size=22)
line(735, 1944, 765, 1944, arrow=True)
module(770, 1905, 175, 77, "SiLU", fill="white", edge="teal_edge", title_size=23)
line(945, 1944, 975, 1944, arrow=True)
module(980, 1905, 352, 77, "Pointwise 1 × 1 × 1 conv", fill="white", edge="teal_edge", title_size=22)
path("M1807 1550 H1830 V1860 H1628", "teal_edge", 2.5, arrow=True)
path("M1332 1944 H1590 V1886 H1628", "teal_edge", 2.5, arrow=True)
node(1653, 1870, "+", "teal_edge", 24, 28)
line(1677, 1870, 1700, 1870, arrow=True)
module(1702, 1830, 145, 81, "γ ·", ("channel-wise",), "teal", "teal_edge", 26, 16)
text(1775, 1938, "γ init 0.01", 19, 600, "teal_deep", "middle")
path("M295 1505 H315 V2085 H1748 V2050 H1750", "teal_edge", 2.6, arrow=True)
text(613, 2065, "identity skip X", 21, 600, "teal_deep")
path("M1775 1911 V2026", "teal_edge", 2.5, arrow=True)
node(1775, 2050, "+", "teal_edge", 24, 28)
line(1775, 2074, 1775, 2106, arrow=True)
text(1775, 2145, "Output  B × C × D × H × W", 23, 700, "teal_deep", "middle")

# PANEL C --------------------------------------------------------------------
rect(1950, 1160, 1895, 1075, "white", "hair", 22, 2)
tag(1979, 1185, "(c)", "orange_edge")
text(2052, 1206, "Gated 3D frequency-aware fusion", 35, 700)
text(1995, 1280, "Inputs at the same spatial resolution", 22, 400, "muted")

module(1995, 1338, 160, 82, "H", ("encoder skip",), "orange", "orange_edge", 28, 18)
module(1995, 1811, 160, 82, "L", ("upsampled decoder",), "orange", "orange_edge", 28, 17)
module(2200, 1338, 206, 82, "1 × 1 × 1 conv", ("16 channels",), "white", "orange_edge", 21, 18)
module(2200, 1811, 206, 82, "1 × 1 × 1 conv", ("16 channels",), "white", "orange_edge", 21, 18)
line(2155, 1379, 2197, 1379, arrow=True)
line(2155, 1852, 2197, 1852, arrow=True)
path("M2406 1379 H2460 V1588", "orange_edge", 2.5, arrow=True)
path("M2406 1852 H2460 V1636", "orange_edge", 2.5, arrow=True)
node(2460, 1612, "+", "orange_edge", 24, 27)
line(2484, 1612, 2510, 1612, arrow=True)
module(2515, 1572, 160, 82, "SiLU", ("shared context",), "orange", "orange_edge", 23, 18)

# Four independent branches from shared context: two kernel generators, two gates.
path("M2675 1612 H2705 V1381 H2727", "orange_edge", 2.3, arrow=True)
path("M2675 1612 H2705 V1513 H2727", "orange_edge", 2.3, arrow=True)
path("M2675 1612 H2705 V1760 H2727", "orange_edge", 2.3, arrow=True)
path("M2675 1612 H2705 V1892 H2727", "orange_edge", 2.3, arrow=True)
module(2730, 1338, 240, 84, "K_H generator", ("27 weights / voxel",), "orange", "orange_edge", 22, 18)
module(2730, 1470, 240, 84, "sigmoid gate g_H", fill="white", edge="orange_edge", title_size=20)
module(2730, 1717, 240, 84, "K_L generator", ("27 weights / voxel",), "orange", "orange_edge", 22, 18)
module(2730, 1849, 240, 84, "sigmoid gate g_L", fill="white", edge="orange_edge", title_size=20)
text(2850, 1990, "K: softmax × 3D Hamming window, then renormalize", 20, 400, "orange_deep", "middle")
text(2850, 2023, "Spatially adaptive weights, shared across channels", 20, 400, "orange_deep", "middle")

# High-resolution skip: adaptive low-pass, subtract to obtain high-pass, gate, residual add.
module(3030, 1338, 165, 82, "K_H * H", ("low-pass",), "white", "orange_edge", 22, 17)
module(3232, 1338, 165, 82, "H − K_H * H", ("high-pass residual",), "white", "orange_edge", 20, 16)
node(3436, 1379, "×", "orange_edge", 23, 26)
node(3595, 1379, "+", "orange_edge", 23, 26)
text(3755, 1379, "H′", 31, 700, "orange_deep", "middle")
line(2970, 1380, 3027, 1380, arrow=True)
line(3195, 1379, 3229, 1379, arrow=True)
line(3397, 1379, 3412, 1379, arrow=True)
line(3460, 1379, 3570, 1379, arrow=True)
line(3619, 1379, 3715, 1379, arrow=True)
path("M2970 1513 H3436 V1353", "orange_edge", 2.3, arrow=True)
path("M2155 1379 V1305 H3595 V1353", "orange_edge", 2.1, arrow=True)
path("M3000 1305 V1379 H3027", "orange_edge", 2.1, arrow=True)
path("M3215 1305 V1379 H3229", "orange_edge", 2.1, arrow=True)
text(3638, 1291, "original H", 18, 400, "muted", "end")

# Decoder: adaptive smoothing, subtract original L, gate, residual add.
module(3030, 1717, 165, 82, "K_L * L", ("smoothing",), "white", "orange_edge", 22, 17)
module(3232, 1717, 165, 82, "K_L * L − L", ("low-pass residual",), "white", "orange_edge", 19, 16)
node(3436, 1758, "×", "orange_edge", 23, 26)
node(3595, 1758, "+", "orange_edge", 23, 26)
text(3755, 1758, "L′", 31, 700, "orange_deep", "middle")
line(2970, 1759, 3027, 1759, arrow=True)
line(3195, 1758, 3229, 1758, arrow=True)
line(3397, 1758, 3412, 1758, arrow=True)
line(3460, 1758, 3570, 1758, arrow=True)
line(3619, 1758, 3715, 1758, arrow=True)
path("M2970 1892 H3436 V1734", "orange_edge", 2.3, arrow=True)
path("M2155 1852 V2080 H3595 V1783", "orange_edge", 2.1, arrow=True)
path("M3000 2080 V1758 H3027", "orange_edge", 2.1, arrow=True)
path("M3215 2080 V1758 H3229", "orange_edge", 2.1, arrow=True)
text(3638, 2092, "original L", 18, 400, "muted", "end")

text(3110, 2144, "L′ = L + gL ⊙ (KL * L − L)", 23, 600, "orange_deep", "middle")
text(3485, 2144, "H′ = H + gH ⊙ (H − KH * H)", 23, 600, "orange_deep", "middle")
text(2010, 2199, "* denotes spatially adaptive 3D filtering. H′ and L′ remain separate until concatenation in panel (a).", 21, 400, "muted")

add('</svg>')
OUT.write_text("\n".join(parts), encoding="utf-8")
print(OUT)
