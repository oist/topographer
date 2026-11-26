"""Generate the introductory biological-workflow overview as a 1:1 presentation slide and SVG.

The figure illustrates the chain from sequence-fitness measurements to multiscale
landscape structure:
1. Sequence-Fitness Measurements (data table)
2. Scalar Fitness Network (height-deformed Hamming cube)
3. Split Tree (superlevel merge tree with peak branches and saddles)
4. Persistence & Simplification (persistence threshold filtering noise)
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np

HTML_OUTPUT = Path(__file__).parents[1] / "docs" / "figures" / "biological_workflow.html"
SVG_OUTPUT = Path(__file__).parents[1] / "docs" / "figures" / "biological_workflow.svg"

# Canonical 3-peak 3-locus landscape
GENOTYPES = ["011", "000", "110", "111", "001", "010", "101", "100"]
FITNESS = {
    "011": 1.00,  # Dominant global peak
    "000": 0.70,  # Robust secondary peak
    "110": 0.50,  # Minor noise peak
    "111": 0.40,  # Split saddle 1 (where 110 merges)
    "001": 0.35,  # Split saddle 2 (where 000 merges)
    "010": 0.30,  # Valley
    "101": 0.20,  # Valley
    "100": 0.15,  # Global minimum (root)
}

ROLES = {
    "011": "Dominant Peak",
    "000": "Robust Peak",
    "110": "Noise Peak",
    "111": "Split Saddle 1",
    "001": "Split Saddle 2",
    "010": "Valley",
    "101": "Valley",
    "100": "Global Minimum",
}

EDGES = [
    ("000", "100"), ("001", "101"), ("010", "110"), ("011", "111"),
    ("000", "010"), ("001", "011"), ("100", "110"), ("101", "111"),
    ("000", "001"), ("010", "011"), ("100", "101"), ("110", "111"),
]

VIRIDIS_HEX = {
    "011": "#fde725",
    "000": "#44bf70",
    "110": "#21918c",
    "111": "#2a788e",
    "001": "#2f6c8e",
    "010": "#355f8d",
    "101": "#414487",
    "100": "#463480",
}


def build_html_slide() -> str:
    """Construct the self-contained 1:1 square presentation slide HTML."""
    rows_html = []
    for seq in GENOTYPES:
        f = FITNESS[seq]
        role = ROLES[seq]
        color = VIRIDIS_HEX[seq]
        bar_width = int(f * 70)
        badge_class = "role-valley"
        if "Dominant" in role:
            badge_class = "role-peak-global"
        elif "Robust" in role:
            badge_class = "role-peak-robust"
        elif "Noise" in role:
            badge_class = "role-peak-noise"
        elif "Saddle" in role:
            badge_class = "role-saddle"

        rows_html.append(f"""            <tr>
              <td><span class="seq-code">{seq}</span></td>
              <td>
                <div class="fitness-bar-container">
                  <span style="font-weight: 600; width: 28px;">{f:.2f}</span>
                  <div class="fitness-bar" style="width: {bar_width}px; background: {color};"></div>
                </div>
              </td>
              <td><span class="role-badge {badge_class}">{role}</span></td>
            </tr>""")
    table_rows = "\n".join(rows_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Topographer: From Sequence-Fitness Measurements to Multiscale Landscape Structure</title>
  <style>
    :root {{
      --bg: #F8FAFC;
      --card-bg: #FFFFFF;
      --border: #E2E8F0;
      --text-main: #0F172A;
      --text-muted: #64748B;
      --accent-blue: #0284C7;
      --accent-indigo: #4F46E5;
      --accent-orange: #EA580C;
      --accent-emerald: #059669;
      --accent-rose: #EF4444;
      
      --c-100: #fde725;
      --c-070: #44bf70;
      --c-050: #21918c;
      --c-040: #2a788e;
      --c-035: #2f6c8e;
      --c-030: #355f8d;
      --c-020: #414487;
      --c-015: #463480;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: #E2E8F0;
      color: var(--text-main);
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
    }}

    .slide-container {{
      width: 1000px;
      height: 1000px;
      background: var(--bg);
      border-radius: 16px;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
      display: flex;
      flex-direction: column;
      padding: 28px;
      position: relative;
      overflow: hidden;
    }}

    header {{
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      border-bottom: 1px solid var(--border);
      padding-bottom: 14px;
    }}

    .title-area h1 {{
      font-size: 20px;
      font-weight: 700;
      color: var(--text-main);
      letter-spacing: -0.02em;
    }}

    .title-area p {{
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 3px;
    }}

    .header-tag {{
      font-size: 11px;
      font-weight: 600;
      color: var(--accent-indigo);
      background: #EEF2FF;
      border: 1px solid #C7D2FE;
      padding: 4px 10px;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .grid-layout {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr;
      gap: 18px;
      flex: 1;
      min-height: 0;
    }}

    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
      position: relative;
    }}

    .card-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }}

    .step-badge {{
      width: 24px;
      height: 24px;
      border-radius: 6px;
      background: var(--accent-indigo);
      color: white;
      font-size: 12px;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }}

    .card-header h2 {{
      font-size: 14px;
      font-weight: 700;
      color: var(--text-main);
    }}

    .card-header span.subtitle {{
      font-size: 11px;
      color: var(--text-muted);
      margin-left: auto;
    }}

    .card-body {{
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      position: relative;
      min-height: 0;
    }}

    .flow-arrow {{
      position: absolute;
      font-weight: bold;
      color: var(--accent-indigo);
      opacity: 0.85;
      pointer-events: none;
      z-index: 10;
    }}
    .arrow-1-2 {{
      top: 245px;
      left: calc(50% - 10px);
      font-size: 20px;
    }}
    .arrow-2-3 {{
      top: 485px;
      right: 48px;
      font-size: 20px;
      transform: rotate(90deg);
    }}
    .arrow-3-4 {{
      bottom: 245px;
      left: calc(50% - 10px);
      font-size: 20px;
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11.5px;
    }}

    .data-table th {{
      text-align: left;
      padding: 6px 8px;
      color: var(--text-muted);
      font-weight: 600;
      border-bottom: 1px solid var(--border);
      font-size: 10.5px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}

    .data-table td {{
      padding: 5px 8px;
      border-bottom: 1px solid #F1F5F9;
      color: var(--text-main);
    }}

    .seq-code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-weight: 600;
      background: #F1F5F9;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 11px;
    }}

    .fitness-bar-container {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .fitness-bar {{
      height: 7px;
      border-radius: 4px;
    }}

    .role-badge {{
      font-size: 9.5px;
      font-weight: 600;
      padding: 2px 6px;
      border-radius: 4px;
      white-space: nowrap;
    }}
    .role-peak-global {{ background: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }}
    .role-peak-robust {{ background: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; }}
    .role-peak-noise  {{ background: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }}
    .role-saddle      {{ background: #E0E7FF; color: #4338CA; border: 1px solid #C7D2FE; }}
    .role-valley      {{ background: #F3F4F6; color: #4B5563; border: 1px solid #E5E7EB; }}

    .card-footer-note {{
      margin-top: 6px;
      font-size: 10.5px;
      color: var(--text-muted);
      background: #F8FAFC;
      padding: 6px 10px;
      border-radius: 6px;
      border-left: 3px solid var(--accent-indigo);
      width: 100%;
    }}

    svg text {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      user-select: none;
    }}

    footer {{
      margin-top: 14px;
      border-top: 1px solid var(--border);
      padding-top: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .legend-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 11px;
      color: var(--text-muted);
    }}

    .colorbar-wrapper {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .colorbar-gradient {{
      width: 180px;
      height: 10px;
      border-radius: 5px;
      background: linear-gradient(to right, #463480, #414487, #355f8d, #2a788e, #21918c, #44bf70, #fde725);
      border: 1px solid #CBD5E1;
    }}

    .caption {{
      font-size: 10.5px;
      color: var(--text-muted);
      line-height: 1.45;
    }}
  </style>
</head>
<body>

<div class="slide-container">
  <header>
    <div class="title-area">
      <h1>From Sequence-Fitness Measurements to Multiscale Landscape Structure</h1>
      <p>Topographer workflow: reconstructing fitness landscapes and filtering topological noise via persistence</p>
    </div>
    <div class="header-tag">Topological Data Analysis</div>
  </header>

  <div class="flow-arrow arrow-1-2">→</div>
  <div class="flow-arrow arrow-2-3">→</div>
  <div class="flow-arrow arrow-3-4">→</div>

  <div class="grid-layout">

    <!-- CARD 1: MEASUREMENTS -->
    <div class="card">
      <div class="card-header">
        <div class="step-badge">1</div>
        <h2>Sequence-Fitness Measurements</h2>
        <span class="subtitle">Genotype Table</span>
      </div>
      <div class="card-body">
        <table class="data-table">
          <thead>
            <tr>
              <th>Sequence</th>
              <th>Fitness f(v)</th>
              <th>Landscape Role</th>
            </tr>
          </thead>
          <tbody>
{table_rows}
          </tbody>
        </table>
      </div>
      <div class="card-footer-note">
        Edges connect sequences differing by a single mutation (d<sub>H</sub> = 1).
      </div>
    </div>

    <!-- CARD 2: SCALAR NETWORK -->
    <div class="card">
      <div class="card-header">
        <div class="step-badge">2</div>
        <h2>Scalar Fitness Network</h2>
        <span class="subtitle">Hamming Cube Graph</span>
      </div>
      <div class="card-body">
        <svg viewBox="0 0 420 310" width="100%" height="100%">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#64748B"/>
            </marker>
          </defs>

          <!-- Fitness height axis guide -->
          <line x1="28" y1="270" x2="28" y2="40" stroke="#CBD5E1" stroke-width="1.2" marker-end="url(#arrow)"/>
          <text x="22" y="32" font-size="10" font-weight="600" fill="#64748B" text-anchor="middle">Fitness</text>
          <text x="18" y="55" font-size="8.5" fill="#94A3B8" text-anchor="end">1.0</text>
          <text x="18" y="160" font-size="8.5" fill="#94A3B8" text-anchor="end">0.5</text>
          <text x="18" y="270" font-size="8.5" fill="#94A3B8" text-anchor="end">0.0</text>

          <!-- 12 Hamming Edges -->
          <line x1="120" y1="130" x2="270" y2="260" stroke="#94A3B8" stroke-width="1.2"/>
          <line x1="200" y1="150" x2="350" y2="245" stroke="#94A3B8" stroke-width="1.2"/>
          <line x1="120" y1="240" x2="270" y2="180" stroke="#94A3B8" stroke-width="1.2"/>
          <line x1="200" y1="70"  x2="350" y2="190" stroke="#94A3B8" stroke-width="1.5" stroke-dasharray="3,3"/>

          <line x1="120" y1="130" x2="120" y2="240" stroke="#94A3B8" stroke-width="1.2"/>
          <line x1="200" y1="150" x2="200" y2="70"  stroke="#94A3B8" stroke-width="1.5"/>
          <line x1="270" y1="260" x2="270" y2="180" stroke="#94A3B8" stroke-width="1.2"/>
          <line x1="350" y1="245" x2="350" y2="190" stroke="#94A3B8" stroke-width="1.2"/>

          <line x1="120" y1="130" x2="200" y2="150" stroke="#94A3B8" stroke-width="1.2"/>
          <line x1="120" y1="240" x2="200" y2="70"  stroke="#94A3B8" stroke-width="1.5"/>
          <line x1="270" y1="260" x2="350" y2="245" stroke="#94A3B8" stroke-width="1.2"/>
          <line x1="270" y1="180" x2="350" y2="190" stroke="#94A3B8" stroke-width="1.2"/>

          <!-- Vertices -->
          <circle cx="270" cy="260" r="7" fill="var(--c-015)" stroke="#0F172A" stroke-width="1.2"/>
          <text x="270" y="277" font-size="9" font-family="monospace" fill="#0F172A" text-anchor="middle">100 (0.15)</text>

          <circle cx="350" cy="245" r="7" fill="var(--c-020)" stroke="#0F172A" stroke-width="1.2"/>
          <text x="358" y="260" font-size="9" font-family="monospace" fill="#0F172A" text-anchor="start">101</text>

          <circle cx="120" cy="240" r="7" fill="var(--c-030)" stroke="#0F172A" stroke-width="1.2"/>
          <text x="110" y="255" font-size="9" font-family="monospace" fill="#0F172A" text-anchor="end">010 (0.30)</text>

          <circle cx="200" cy="150" r="7.5" fill="var(--c-035)" stroke="#0F172A" stroke-width="1.2"/>
          <text x="190" y="145" font-size="9" font-family="monospace" fill="#0F172A" text-anchor="end">001 (0.35)</text>

          <circle cx="350" cy="190" r="7.5" fill="var(--c-040)" stroke="#0F172A" stroke-width="1.2"/>
          <text x="362" y="194" font-size="9" font-family="monospace" fill="#0F172A" text-anchor="start">111 (0.40)</text>

          <circle cx="270" cy="180" r="8.5" fill="var(--c-050)" stroke="#B91C1C" stroke-width="2"/>
          <text x="270" y="166" font-size="9.5" font-weight="bold" font-family="monospace" fill="#B91C1C" text-anchor="middle">110 (Peak 3)</text>

          <circle cx="120" cy="130" r="9" fill="var(--c-070)" stroke="#15803D" stroke-width="2"/>
          <text x="110" y="122" font-size="9.5" font-weight="bold" font-family="monospace" fill="#15803D" text-anchor="end">000 (Peak 2)</text>

          <circle cx="200" cy="70" r="10.5" fill="var(--c-100)" stroke="#B45309" stroke-width="2.2"/>
          <text x="200" y="52" font-size="10" font-weight="bold" font-family="monospace" fill="#B45309" text-anchor="middle">★ 011 (Global Peak 1.00)</text>
        </svg>
      </div>
      <div class="card-footer-note">
        Vertical axis & color encode fitness; 3 distinct peaks emerge across the cube.
      </div>
    </div>

    <!-- CARD 3: SPLIT TREE -->
    <div class="card">
      <div class="card-header">
        <div class="step-badge">3</div>
        <h2>Split Tree (Merge Tree)</h2>
        <span class="subtitle">Superlevel Sweep</span>
      </div>
      <div class="card-body">
        <svg viewBox="0 0 420 310" width="100%" height="100%">
          <line x1="36" y1="40" x2="36" y2="280" stroke="#CBD5E1" stroke-width="1.2"/>
          <polygon points="32,278 36,288 40,278" fill="#64748B"/>
          <text x="36" y="26" font-size="10" font-weight="600" fill="#64748B" text-anchor="middle">Threshold τ ↓</text>
          
          <line x1="32" y1="45" x2="36" y2="45" stroke="#64748B" stroke-width="1.2"/>
          <text x="28" y="48" font-size="9" fill="#64748B" text-anchor="end">1.00</text>

          <line x1="32" y1="110" x2="36" y2="110" stroke="#64748B" stroke-width="1.2"/>
          <text x="28" y="113" font-size="9" fill="#64748B" text-anchor="end">0.70</text>

          <line x1="32" y1="150" x2="36" y2="150" stroke="#64748B" stroke-width="1.2"/>
          <text x="28" y="153" font-size="9" fill="#64748B" text-anchor="end">0.50</text>

          <line x1="32" y1="172" x2="36" y2="172" stroke="#64748B" stroke-width="1.2"/>
          <text x="28" y="175" font-size="9" fill="#64748B" text-anchor="end">0.40</text>

          <line x1="32" y1="184" x2="36" y2="184" stroke="#64748B" stroke-width="1.2"/>
          <text x="28" y="187" font-size="9" fill="#64748B" text-anchor="end">0.35</text>

          <line x1="32" y1="230" x2="36" y2="230" stroke="#64748B" stroke-width="1.2"/>
          <text x="28" y="233" font-size="9" fill="#64748B" text-anchor="end">0.15</text>

          <line x1="36" y1="172" x2="380" y2="172" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="2,2"/>
          <line x1="36" y1="184" x2="380" y2="184" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="2,2"/>

          <!-- Trunk -->
          <line x1="220" y1="45" x2="220" y2="230" stroke="#0284C7" stroke-width="3" stroke-linecap="round"/>

          <!-- Peak 110 Branch -->
          <path d="M 320 150 L 320 166 Q 320 172 312 172 L 220 172" fill="none" stroke="#EA580C" stroke-width="2.2" stroke-linecap="round"/>

          <!-- Peak 000 Branch -->
          <path d="M 120 110 L 120 178 Q 120 184 128 184 L 220 184" fill="none" stroke="#059669" stroke-width="2.6" stroke-linecap="round"/>

          <!-- Peak Vertices -->
          <polygon points="220,38 226,48 214,48" fill="var(--c-100)" stroke="#0F172A" stroke-width="1.4"/>
          <text x="220" y="32" font-size="9.5" font-weight="bold" font-family="monospace" fill="#0F172A" text-anchor="middle">011 (1.00)</text>

          <polygon points="120,103 126,113 114,113" fill="var(--c-070)" stroke="#0F172A" stroke-width="1.4"/>
          <text x="120" y="97" font-size="9.5" font-weight="bold" font-family="monospace" fill="#0F172A" text-anchor="middle">000 (0.70)</text>

          <polygon points="320,143 326,153 314,153" fill="var(--c-050)" stroke="#0F172A" stroke-width="1.4"/>
          <text x="320" y="137" font-size="9.5" font-weight="bold" font-family="monospace" fill="#0F172A" text-anchor="middle">110 (0.50)</text>

          <!-- Saddles -->
          <polygon points="220,166 226,172 220,178 214,172" fill="#FFFFFF" stroke="#EA580C" stroke-width="1.8"/>
          <text x="232" y="169" font-size="8.5" font-family="monospace" fill="#EA580C">saddle 111 (0.40)</text>

          <polygon points="220,178 226,184 220,190 214,184" fill="#FFFFFF" stroke="#059669" stroke-width="1.8"/>
          <text x="232" y="190" font-size="8.5" font-family="monospace" fill="#059669">saddle 001 (0.35)</text>

          <!-- Root -->
          <circle cx="220" cy="230" r="5" fill="var(--c-015)" stroke="#0F172A" stroke-width="1.4"/>
          <text x="220" y="246" font-size="9" font-family="monospace" fill="#0F172A" text-anchor="middle">root 100 (0.15)</text>

          <!-- Span annotations -->
          <path d="M 335 150 L 345 150 L 345 172 L 335 172" fill="none" stroke="#EA580C" stroke-width="1"/>
          <text x="350" y="163" font-size="8.5" font-weight="600" fill="#EA580C">span = 0.10</text>

          <path d="M 105 110 L 95 110 L 95 184 L 105 184" fill="none" stroke="#059669" stroke-width="1"/>
          <text x="90" y="150" font-size="8.5" font-weight="600" fill="#059669" text-anchor="end">span = 0.35</text>
        </svg>
      </div>
      <div class="card-footer-note">
        Elder rule: at each saddle, the younger component dies; the elder component survives.
      </div>
    </div>

    <!-- CARD 4: PERSISTENCE & SIMPLIFICATION -->
    <div class="card">
      <div class="card-header">
        <div class="step-badge">4</div>
        <h2>Persistence & Simplification</h2>
        <span class="subtitle">Thresholding Noise (τ = 0.20)</span>
      </div>
      <div class="card-body" style="flex-direction: row; gap: 12px; align-items: stretch;">
        <div style="flex: 1; display: flex; flex-direction: column; justify-content: space-between; border-right: 1px dashed var(--border); padding-right: 12px;">
          <div style="font-size: 11px; font-weight: 700; color: var(--text-main); margin-bottom: 8px;">
            Feature Persistence (Birth − Death)
          </div>

          <div>
            <div style="display: flex; justify-content: space-between; font-size: 9.5px; margin-bottom: 2px;">
              <span style="font-family: monospace; font-weight: 600;">011 (Global Peak)</span>
              <span style="color: var(--accent-blue); font-weight: 700;">pers = 0.85</span>
            </div>
            <div style="background: #E2E8F0; height: 14px; border-radius: 4px; overflow: hidden; position: relative;">
              <div style="width: 85%; height: 100%; background: var(--accent-blue);"></div>
            </div>
          </div>

          <div>
            <div style="display: flex; justify-content: space-between; font-size: 9.5px; margin-bottom: 2px;">
              <span style="font-family: monospace; font-weight: 600;">000 (Robust Peak)</span>
              <span style="color: var(--accent-emerald); font-weight: 700;">pers = 0.35</span>
            </div>
            <div style="background: #E2E8F0; height: 14px; border-radius: 4px; overflow: hidden; position: relative;">
              <div style="width: 35%; height: 100%; background: var(--accent-emerald);"></div>
            </div>
          </div>

          <div>
            <div style="display: flex; justify-content: space-between; font-size: 9.5px; margin-bottom: 2px;">
              <span style="font-family: monospace; font-weight: 600;">110 (Noise Peak)</span>
              <span style="color: var(--accent-orange); font-weight: 700;">pers = 0.10</span>
            </div>
            <div style="background: #E2E8F0; height: 14px; border-radius: 4px; overflow: hidden; position: relative;">
              <div style="width: 10%; height: 100%; background: var(--accent-orange);"></div>
            </div>
          </div>

          <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 6px 8px; font-size: 9.5px; color: #92400E;">
            <strong>Selected Threshold τ = 0.20:</strong><br>
            • pers ≥ 0.20: <strong>Retained</strong> (2 Peaks)<br>
            • pers &lt; 0.20: <strong>Pruned Noise</strong> (1 Peak)
          </div>
        </div>

        <div style="flex: 1.1; display: flex; flex-direction: column; align-items: center;">
          <div style="font-size: 11px; font-weight: 700; color: var(--text-main); margin-bottom: 4px;">
            Simplified Split Tree (T<sup>simp</sup>)
          </div>
          
          <svg viewBox="0 0 210 190" width="100%" height="150">
            <line x1="110" y1="20" x2="110" y2="160" stroke="#0284C7" stroke-width="3" stroke-linecap="round"/>
            <path d="M 40 65 L 40 120 Q 40 126 48 126 L 110 126" fill="none" stroke="#059669" stroke-width="2.6" stroke-linecap="round"/>

            <!-- Pruned branch -->
            <path d="M 180 92 L 180 112 Q 180 117 172 117 L 110 117" fill="none" stroke="#CBD5E1" stroke-width="1.8" stroke-dasharray="3,3"/>
            <line x1="174" y1="88" x2="186" y2="98" stroke="#EF4444" stroke-width="1.5"/>
            <line x1="186" y1="88" x2="174" y2="98" stroke="#EF4444" stroke-width="1.5"/>
            <text x="188" y="94" font-size="8" fill="#EF4444" font-weight="600">pruned</text>

            <polygon points="110,14 115,23 105,23" fill="var(--c-100)" stroke="#0F172A" stroke-width="1.2"/>
            <text x="110" y="10" font-size="8.5" font-weight="bold" font-family="monospace" fill="#0F172A" text-anchor="middle">011</text>

            <polygon points="40,59 45,68 35,68" fill="var(--c-070)" stroke="#0F172A" stroke-width="1.2"/>
            <text x="40" y="55" font-size="8.5" font-weight="bold" font-family="monospace" fill="#0F172A" text-anchor="middle">000</text>

            <polygon points="110,122 114,126 110,130 106,126" fill="#FFFFFF" stroke="#059669" stroke-width="1.5"/>

            <circle cx="110" cy="160" r="4.5" fill="var(--c-015)" stroke="#0F172A" stroke-width="1.2"/>
            <text x="110" y="174" font-size="8.5" font-family="monospace" fill="#0F172A" text-anchor="middle">100</text>
          </svg>

          <div style="font-size: 9.5px; color: var(--accent-emerald); font-weight: 600; text-align: center; margin-top: 2px;">
            ✓ 2 Robust Peaks Retained
          </div>
        </div>
      </div>
      <div class="card-footer-note">
        Pruning low-persistence branches simplifies the landscape summary without changing the underlying measurements.
      </div>
    </div>

  </div>

  <footer>
    <div class="legend-row">
      <div class="colorbar-wrapper">
        <span style="font-weight: 600;">Fitness Scale:</span>
        <span>0.0 (Low)</span>
        <div class="colorbar-gradient"></div>
        <span>1.0 (High)</span>
      </div>
      <div style="font-size: 10px; color: var(--text-muted);">
        Topographer v0.3 • Plan 17 Biological Workflow Specification
      </div>
    </div>
    <div class="caption">
      <strong>Schematic 3-locus binary sequence-fitness landscape.</strong> Network edges connect single-nucleotide mutants (d<sub>H</sub> = 1). The split tree traces connected components across decreasing fitness thresholds. Ordinary persistence quantifies feature prominence (birth − death); thresholding at τ = 0.20 filters minor local fluctuations to recover dominant landscape peaks without modifying the original data.
    </div>
  </footer>
</div>

</body>
</html>
"""
    return html


def build_svg_figure() -> Path:
    """Render the 1:1 vector SVG overview figure using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#FFFFFF")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    palette = {
        "ink": "#0F172A",
        "muted": "#64748B",
        "line": "#CBD5E1",
        "card_border": "#E2E8F0",
        "blue": "#0284C7",
        "indigo": "#4F46E5",
        "orange": "#EA580C",
        "green": "#059669",
        "paper": "#FFFFFF",
    }

    # Main Slide Container Card
    slide = FancyBboxPatch(
        (0.015, 0.015), 0.97, 0.97,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        facecolor="#F8FAFC", edgecolor=palette["card_border"], linewidth=1.5
    )
    ax.add_patch(slide)

    # Header
    ax.text(0.04, 0.955, "From Sequence-Fitness Measurements to Multiscale Landscape Structure",
            fontsize=13, fontweight="bold", color=palette["ink"], va="top")
    ax.text(0.04, 0.932, "Topographer workflow: reconstructing fitness landscapes and filtering topological noise via persistence",
            fontsize=8.5, color=palette["muted"], va="top")
    ax.plot([0.04, 0.96], [0.915, 0.915], color=palette["card_border"], lw=1)

    # 4 Cards Helper
    def draw_card(cx, cy, cw, ch, step_num, title, subtitle):
        card_patch = FancyBboxPatch(
            (cx, cy), cw, ch,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=palette["paper"], edgecolor=palette["card_border"], linewidth=1.2, zorder=2
        )
        ax.add_patch(card_patch)
        badge = FancyBboxPatch(
            (cx + 0.015, cy + ch - 0.045), 0.024, 0.024,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            facecolor=palette["indigo"], edgecolor="none", zorder=3
        )
        ax.add_patch(badge)
        ax.text(cx + 0.027, cy + ch - 0.033, str(step_num), color="#FFFFFF",
                fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(cx + 0.045, cy + ch - 0.033, title, color=palette["ink"],
                fontsize=9.5, fontweight="bold", va="center", zorder=3)
        ax.text(cx + cw - 0.015, cy + ch - 0.033, subtitle, color=palette["muted"],
                fontsize=7.5, ha="right", va="center", zorder=3)

    # Card 1: Measurements (0.04, 0.50, 0.44, 0.39)
    draw_card(0.04, 0.50, 0.44, 0.39, 1, "Sequence-Fitness Measurements", "Genotype Table")
    # Draw table rows
    y_start = 0.835
    ax.text(0.065, y_start, "Sequence", fontsize=7.5, fontweight="bold", color=palette["muted"], zorder=3)
    ax.text(0.18, y_start, "Fitness f(v)", fontsize=7.5, fontweight="bold", color=palette["muted"], zorder=3)
    ax.text(0.33, y_start, "Landscape Role", fontsize=7.5, fontweight="bold", color=palette["muted"], zorder=3)
    ax.plot([0.06, 0.45], [y_start - 0.01, y_start - 0.01], color=palette["card_border"], lw=0.8, zorder=3)

    for i, seq in enumerate(GENOTYPES):
        ry = y_start - 0.028 * (i + 1)
        f = FITNESS[seq]
        role = ROLES[seq]
        col = VIRIDIS_HEX[seq]
        ax.text(0.065, ry, seq, fontfamily="monospace", fontsize=7.5, fontweight="bold", color=palette["ink"], zorder=3)
        ax.text(0.18, ry, f"{f:.2f}", fontsize=7.5, fontweight="bold", color=palette["ink"], zorder=3)
        ax.plot([0.22, 0.22 + f * 0.07], [ry + 0.003, ry + 0.003], color=col, lw=3.5, solid_capstyle="round", zorder=3)
        ax.text(0.33, ry, role, fontsize=7.0, color=palette["muted"], zorder=3)

    # Card 2: Scalar Network (0.52, 0.50, 0.44, 0.39)
    draw_card(0.52, 0.50, 0.44, 0.39, 2, "Scalar Fitness Network", "Hamming Cube")
    net_pts = {
        "000": (0.59, 0.72), "001": (0.69, 0.69), "010": (0.59, 0.58), "011": (0.69, 0.81),
        "100": (0.78, 0.55), "101": (0.88, 0.58), "110": (0.78, 0.67), "111": (0.88, 0.64),
    }
    for u, v in EDGES:
        p1, p2 = net_pts[u], net_pts[v]
        ls = "--" if (u, v) in [("011", "111"), ("111", "011")] else "-"
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=palette["line"], lw=1.2, ls=ls, zorder=3)
    for seq, pt in net_pts.items():
        f = FITNESS[seq]
        col = VIRIDIS_HEX[seq]
        sz = 80 if seq in ["011", "000", "110"] else 45
        ec = palette["ink"]
        ax.scatter(pt[0], pt[1], s=sz, color=col, edgecolor=ec, linewidth=1.0, zorder=4)
        lbl = f"{seq} ({f:.2f})"
        if seq == "011":
            ax.text(pt[0], pt[1] + 0.016, "★ 011 (Peak 1.00)", fontsize=7.5, fontweight="bold", ha="center", color="#B45309", zorder=5)
        elif seq == "000":
            ax.text(pt[0] - 0.012, pt[1] + 0.008, "000 (Peak)", fontsize=7.0, fontweight="bold", ha="right", color=palette["green"], zorder=5)
        elif seq == "110":
            ax.text(pt[0] + 0.012, pt[1] + 0.008, "110 (Peak)", fontsize=7.0, fontweight="bold", ha="left", color=palette["orange"], zorder=5)
        elif seq == "100":
            ax.text(pt[0], pt[1] - 0.016, "100 (Min)", fontsize=6.5, ha="center", color=palette["muted"], zorder=5)

    # Card 3: Split Tree (0.04, 0.08, 0.44, 0.39)
    draw_card(0.04, 0.08, 0.44, 0.39, 3, "Split Tree (Merge Tree)", "Superlevel Sweep")
    # Axis
    ax.plot([0.08, 0.08], [0.13, 0.41], color=palette["line"], lw=1, zorder=3)
    ax.text(0.075, 0.415, "τ ↓", fontsize=7.5, fontweight="bold", color=palette["muted"], ha="center", zorder=3)
    for val, ypos in [(1.0, 0.40), (0.70, 0.33), (0.50, 0.28), (0.40, 0.25), (0.35, 0.235), (0.15, 0.15)]:
        ax.plot([0.075, 0.08], [ypos, ypos], color=palette["muted"], lw=0.8, zorder=3)
        ax.text(0.07, ypos, f"{val:.2f}", fontsize=6.5, ha="right", va="center", color=palette["muted"], zorder=3)

    # Branches
    # 011 trunk
    ax.plot([0.26, 0.26], [0.15, 0.40], color=palette["blue"], lw=3.0, solid_capstyle="round", zorder=3)
    # 000 branch merging at 001 (y=0.235)
    ax.plot([0.16, 0.16, 0.26], [0.33, 0.235, 0.235], color=palette["green"], lw=2.2, solid_capstyle="round", zorder=3)
    # 110 branch merging at 111 (y=0.25)
    ax.plot([0.36, 0.36, 0.26], [0.28, 0.25, 0.25], color=palette["orange"], lw=2.0, solid_capstyle="round", zorder=3)

    # Markers
    ax.scatter([0.26], [0.40], marker="^", s=70, color=VIRIDIS_HEX["011"], edgecolor=palette["ink"], lw=1.2, zorder=4)
    ax.text(0.26, 0.415, "011 (1.00)", fontsize=7.5, fontweight="bold", ha="center", color=palette["ink"], zorder=4)
    ax.scatter([0.16], [0.33], marker="^", s=60, color=VIRIDIS_HEX["000"], edgecolor=palette["ink"], lw=1.2, zorder=4)
    ax.text(0.16, 0.345, "000 (0.70)", fontsize=7.0, fontweight="bold", ha="center", color=palette["ink"], zorder=4)
    ax.scatter([0.36], [0.28], marker="^", s=50, color=VIRIDIS_HEX["110"], edgecolor=palette["ink"], lw=1.2, zorder=4)
    ax.text(0.36, 0.295, "110 (0.50)", fontsize=7.0, fontweight="bold", ha="center", color=palette["ink"], zorder=4)

    # Saddles
    ax.scatter([0.26], [0.25], marker="D", s=40, color=palette["paper"], edgecolor=palette["orange"], lw=1.5, zorder=4)
    ax.text(0.275, 0.25, "saddle 111 (0.40)", fontsize=6.5, va="center", color=palette["orange"], zorder=4)
    ax.scatter([0.26], [0.235], marker="D", s=40, color=palette["paper"], edgecolor=palette["green"], lw=1.5, zorder=4)
    ax.text(0.275, 0.235, "saddle 001 (0.35)", fontsize=6.5, va="center", color=palette["green"], zorder=4)
    # Root
    ax.scatter([0.26], [0.15], marker="o", s=35, color=VIRIDIS_HEX["100"], edgecolor=palette["ink"], lw=1.2, zorder=4)
    ax.text(0.26, 0.135, "root 100", fontsize=7.0, ha="center", color=palette["muted"], zorder=4)

    # Card 4: Persistence & Simplification (0.52, 0.08, 0.44, 0.39)
    draw_card(0.52, 0.08, 0.44, 0.39, 4, "Persistence & Simplification", "Threshold τ = 0.20")
    # Left: Persistence Lifetimes
    ax.text(0.54, 0.41, "Feature Persistence", fontsize=8.0, fontweight="bold", color=palette["ink"], zorder=3)
    bars = [
        ("011 (Global Peak)", 0.85, palette["blue"], 0.36),
        ("000 (Robust Peak)", 0.35, palette["green"], 0.31),
        ("110 (Noise Peak)",  0.10, palette["orange"], 0.26),
    ]
    for label, pers, col, py in bars:
        ax.text(0.54, py + 0.015, label, fontsize=6.8, fontfamily="monospace", color=palette["ink"], zorder=3)
        ax.text(0.70, py + 0.015, f"pers = {pers:.2f}", fontsize=6.8, fontweight="bold", ha="right", color=col, zorder=3)
        ax.plot([0.54, 0.70], [py, py], color="#E2E8F0", lw=6, solid_capstyle="round", zorder=3)
        ax.plot([0.54, 0.54 + pers * 0.16], [py, py], color=col, lw=6, solid_capstyle="round", zorder=4)

    # Threshold Line at 0.20 (0.54 + 0.20 * 0.16 = 0.572)
    ax.plot([0.572, 0.572], [0.23, 0.39], color=palette["orange"], lw=1.2, ls="--", zorder=5)
    ax.text(0.572, 0.215, "τ = 0.20", fontsize=7.0, fontweight="bold", color=palette["orange"], ha="center", zorder=5)

    # Right: Simplified Split Tree
    ax.text(0.83, 0.41, "Simplified Tree T^simp", fontsize=8.0, fontweight="bold", ha="center", color=palette["ink"], zorder=3)
    # Trunk
    ax.plot([0.83, 0.83], [0.15, 0.38], color=palette["blue"], lw=3.0, solid_capstyle="round", zorder=3)
    # 000 branch
    ax.plot([0.76, 0.76, 0.83], [0.32, 0.235, 0.235], color=palette["green"], lw=2.2, solid_capstyle="round", zorder=3)
    # Pruned 110 branch
    ax.plot([0.90, 0.90, 0.83], [0.28, 0.25, 0.25], color="#CBD5E1", lw=1.5, ls="--", zorder=3)
    ax.text(0.91, 0.28, "✕ pruned", fontsize=6.5, fontweight="bold", color="#EF4444", zorder=4)

    ax.scatter([0.83], [0.38], marker="^", s=60, color=VIRIDIS_HEX["011"], edgecolor=palette["ink"], lw=1.2, zorder=4)
    ax.scatter([0.76], [0.32], marker="^", s=50, color=VIRIDIS_HEX["000"], edgecolor=palette["ink"], lw=1.2, zorder=4)
    ax.scatter([0.83], [0.235], marker="D", s=35, color=palette["paper"], edgecolor=palette["green"], lw=1.5, zorder=4)
    ax.scatter([0.83], [0.15], marker="o", s=30, color=VIRIDIS_HEX["100"], edgecolor=palette["ink"], lw=1.2, zorder=4)
    ax.text(0.83, 0.12, "✓ 2 Robust Peaks", fontsize=7.5, fontweight="bold", color=palette["green"], ha="center", zorder=4)

    # Flow Arrows
    for (start, end) in [((0.485, 0.70), (0.515, 0.70)), ((0.74, 0.495), (0.74, 0.475)), ((0.485, 0.27), (0.515, 0.27))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=palette["indigo"], lw=1.5, zorder=10))

    # Footer Caption
    ax.plot([0.04, 0.96], [0.065, 0.065], color=palette["card_border"], lw=1)
    ax.text(0.04, 0.045, "Schematic 3-locus binary sequence-fitness landscape. Network edges connect single-nucleotide mutants (dH = 1). The split tree traces connected",
            fontsize=7.2, color=palette["muted"], va="top")
    ax.text(0.04, 0.030, "components across decreasing fitness thresholds. Ordinary persistence quantifies feature prominence; thresholding at τ = 0.20 filters minor local",
            fontsize=7.2, color=palette["muted"], va="top")
    ax.text(0.04, 0.015, "fluctuations to recover dominant landscape peaks without modifying the original input measurements.",
            fontsize=7.2, color=palette["muted"], va="top")

    fig.savefig(SVG_OUTPUT, format="svg", bbox_inches="tight")
    plt.close(fig)
    return SVG_OUTPUT


def generate_all() -> None:
    """Generate both the HTML presentation slide and vector SVG."""
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Write HTML presentation slide
    html_content = build_html_slide()
    HTML_OUTPUT.write_text(html_content, encoding="utf-8")
    print(f"Generated HTML slide: {HTML_OUTPUT}")

    # 2. Write SVG figure
    svg_path = build_svg_figure()
    print(f"Generated SVG figure: {svg_path}")


if __name__ == "__main__":
    generate_all()

