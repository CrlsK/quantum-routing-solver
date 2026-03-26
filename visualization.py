"""
visualization.py â QCentroid additional output generator
Generates self-contained HTML visualisations in ./additional_output/

Files produced on every run:
  additional_output/route_map.html        â SVG map of depot, customers & routes
  additional_output/solution_dashboard.html â KPI dashboard with charts & tables
"""
import os
import math
import logging

logger = logging.getLogger("qcentroid-user-log")

_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    "#c0392b", "#2980b9", "#16a085", "#8e44ad",
]


def generate_visualizations(input_data: dict, result: dict) -> None:
    """Entry point: write all visualisation files to additional_output/."""
    try:
        os.makedirs("additional_output", exist_ok=True)
        depot     = input_data.get("depot", {})
        customers = input_data.get("customers", [])
        routes    = result.get("routes", [])

        with open("additional_output/route_map.html", "w", encoding="utf-8") as fh:
            fh.write(_route_map_html(depot, customers, routes))

        with open("additional_output/solution_dashboard.html", "w", encoding="utf-8") as fh:
            fh.write(_dashboard_html(input_data, result))

        logger.info("Visualizations written: route_map.html, solution_dashboard.html")
    except Exception as exc:
        logger.warning(f"Visualization generation skipped: {exc}")


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  Route Map
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _route_map_html(depot: dict, customers: list, routes: list) -> str:
    nodes = [{"id": depot.get("id", "depot"),
              "lat": float(depot.get("lat", 0)),
              "lon": float(depot.get("lon", 0)),
              "demand": 0, "type": "depot"}]
    for c in customers:
        nodes.append({"id": c["id"],
                      "lat": float(c["lat"]), "lon": float(c["lon"]),
                      "demand": c.get("demand", 0), "type": "customer"})
    node_map = {n["id"]: n for n in nodes}

    lats = [n["lat"] for n in nodes]
    lons = [n["lon"] for n in nodes]
    if not lats:
        return "<html><body><p>No location data.</p></body></html>"

    pad = 0.10
    lat_r = max(max(lats) - min(lats), 0.01) * (1 + 2 * pad)
    lon_r = max(max(lons) - min(lons), 0.01) * (1 + 2 * pad)
    min_lat = min(lats) - (lat_r * pad / (1 + 2 * pad))
    min_lon = min(lons) - (lon_r * pad / (1 + 2 * pad))
    W, H = 780, 520

    def proj(lat, lon):
        x = (lon - min_lon) / lon_r * W
        y = H - (lat - min_lat) / lat_r * H
        return round(x, 1), round(y, 1)

    svgs = []

    # Background grid
    for i in range(1, 5):
        gy = int(H * i / 4)
        gx = int(W * i / 4)
        svgs.append(f'<line x1="0" y1="{gy}" x2="{W}" y2="{gy}" stroke="#dfe6e9" stroke-width="1"/>')
        svgs.append(f'<line x1="{gx}" y1="0" x2="{gx}" y2="{H}" stroke="#dfe6e9" stroke-width="1"/>')

    # Routes (polylines + direction arrows)
    for i, route in enumerate(routes):
        col = _COLORS[i % len(_COLORS)]
        seq = route.get("stop_sequence", [])
        pts = []
        for sid in seq:
            n = node_map.get(sid)
            if n:
                x, y = proj(n["lat"], n["lon"])
                pts.append(f"{x},{y}")
        if len(pts) > 1:
            svgs.append(
                f'<polyline points="{" ".join(pts)}" stroke="{col}" '
                f'stroke-width="3" fill="none" opacity="0.85" '
                f'stroke-linejoin="round" stroke-linecap="round"/>')
            # Arrow at each segment midpoint
            for k in range(len(seq) - 1):
                n1 = node_map.get(seq[k])
                n2 = node_map.get(seq[k + 1])
                if n1 and n2:
                    x1, y1 = proj(n1["lat"], n1["lon"])
                    x2, y2 = proj(n2["lat"], n2["lon"])
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    svgs.append(
                        f'<polygon points="-5,-3 5,0 -5,3" fill="{col}" opacity="0.75" '
                        f'transform="translate({mx:.1f},{my:.1f}) rotate({angle:.1f})"/>')

    # Customer nodes
    for n in nodes:
        if n["type"] != "customer":
            continue
        x, y = proj(n["lat"], n["lon"])
        col = "#bdc3c7"
        for i, route in enumerate(routes):
            if n["id"] in route.get("stop_sequence", []):
                col = _COLORS[i % len(_COLORS)]
                break
        svgs.append(
            f'<circle cx="{x}" cy="{y}" r="9" fill="{col}" stroke="white" stroke-width="2.5">'
            f'<title>{n["id"]} | demand={n["demand"]}</title></circle>')
        svgs.append(
            f'<text x="{x}" y="{y - 13}" text-anchor="middle" font-size="9" '
            f'fill="#444" font-family="sans-serif">{n["id"]}</text>')

    # Depot (on top)
    if depot:
        dx, dy = proj(depot.get("lat", 0), depot.get("lon", 0))
        svgs.append(
            f'<rect x="{dx-11}" y="{dy-11}" width="22" height="22" rx="4" '
            f'fill="#2c3e50" stroke="white" stroke-width="2.5">'
            f'<title>DEPOT: {depot.get("id","")}</title></rect>')
        svgs.append(
            f'<text x="{dx}" y="{dy+5}" text-anchor="middle" font-size="9" '
            f'fill="white" font-weight="bold" font-family="sans-serif">D</text>')
        svgs.append(
            f'<text x="{dx}" y="{dy+24}" text-anchor="middle" font-size="10" '
            f'fill="#2c3e50" font-weight="bold" font-family="sans-serif">DEPOT</text>')

    # Legend
    legend_rows = []
    for i, route in enumerate(routes):
        col = _COLORS[i % len(_COLORS)]
        vid = route.get("vehicle_id", f"V{i+1}")
        stops_n = max(len(route.get("stop_sequence", [])) - 2, 0)
        km = route.get("total_km", 0)
        cost_m = route.get("estimated_cost_minutes", 0)
        legend_rows.append(
            f'<tr>'
            f'<td><span style="display:inline-block;width:20px;height:4px;'
            f'background:{col};vertical-align:middle;border-radius:2px"></span></td>'
            f'<td style="padding:2px 8px;font-size:12px"><b>{vid}</b></td>'
            f'<td style="padding:2px 8px;font-size:12px">{stops_n} stops</td>'
            f'<td style="padding:2px 8px;font-size:12px;color:#666">{km:.1f} km</td>'
            f'<td style="padding:2px 8px;font-size:12px;color:#666">{cost_m:.0f} min</td>'
            f'</tr>')

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VRP Route Map</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: #f0f4f8; }}
header {{ background: #2c3e50; color: white; padding: 13px 20px;
          display: flex; align-items: center; gap: 12px; }}
header h2 {{ margin: 0; font-size: 16px; }}
.sub {{ font-size: 12px; opacity: .7; }}
.wrap {{ padding: 16px; display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }}
svg {{ border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,.12); background: #f8fafc; }}
.legend {{ background: white; border-radius: 8px; padding: 14px 16px;
           box-shadow: 0 1px 6px rgba(0,0,0,.10); min-width: 230px; }}
.legend h3 {{ margin: 0 0 10px; font-size: 13px; color: #7f8c8d;
              text-transform: uppercase; letter-spacing: .5px; }}
table {{ border-collapse: collapse; }}
.depot-sq {{ display: inline-block; width: 14px; height: 14px;
             background: #2c3e50; vertical-align: middle; border-radius: 2px; }}
</style></head>
<body>
<header>
  <h2>&#x1F5FA;&#xFE0F; VRP Route Map</h2>
  <span class="sub">{len(customers)} customers &middot; {len(routes)} routes</span>
</header>
<div class="wrap">
  <svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">
    {"".join(svgs)}
  </svg>
  <div class="legend">
    <h3>Vehicle Routes</h3>
    <table>{"".join(legend_rows)}</table>
    <div style="margin-top:12px; display:flex; align-items:center; gap:6px;">
      <span class="depot-sq"></span>
      <span style="font-size:12px;">Depot &mdash; {depot.get("id","")}</span>
    </div>
  </div>
</div>
</body></html>"""


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  Solution Dashboard
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _dashboard_html(input_data: dict, result: dict) -> str:
    routes    = result.get("routes", [])
    n_v       = result.get("total_vehicles_used", len(routes))
    obj       = result.get("objective_value", 0)
    status    = result.get("solution_status", "N/A")
    risk      = result.get("risk_metrics", {})
    on_time   = risk.get("on_time_probability", 1.0)
    cb        = result.get("cost_breakdown", {})
    fuel      = cb.get("fuel_cost_eur", 0)
    travel_t  = cb.get("travel_time_min", obj)
    lateness_t = cb.get("lateness_penalty_min", 0)
    algo      = result.get("algorithm", result.get("solver_type", "N/A"))
    customers = input_data.get("customers", [])
    vehicles  = input_data.get("vehicles", [])
    disruptions = input_data.get("disruptions", [])
    total_km  = round(sum(r.get("total_km", 0) for r in routes), 2)
    total_demand = sum(float(c.get("demand", 1)) for c in customers)
    status_c  = "#27ae60" if status == "optimal" else "#e67e22"
    on_time_pct = f"{on_time * 100:.0f}%"

    # ââ KPI cards ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    kpi_data = [
        (str(n_v),          "Vehicles Used",  "#2980b9"),
        (f"{total_km}",     "Total km",       "#8e44ad"),
        (f"{obj:.0f}",      "Cost (min)",     "#e67e22"),
        (on_time_pct,       "On-Time Rate",   "#27ae60"),
        (f"â¬{fuel:.2f}",    "Fuel Cost",      "#16a085"),
        (str(len(customers)),"Customers",     "#c0392b"),
        (f"{total_demand:.0f}", "Total Demand", "#34495e"),
        (str(len(disruptions)), "Disruptions", "#7f8c8d"),
    ]
    kpis_html = "".join(
        f'<div class="kpi"><div class="v" style="color:{c}">{v}</div>'
        f'<div class="l">{l}</div></div>'
        for v, l, c in kpi_data)

    # ââ Bar chart â distance per vehicle âââââââââââââââââââââââââââââââââââââââ
    max_km = max((r.get("total_km", 0) for r in routes), default=1) or 1
    bw, gap, ch = 52, 12, 150
    bars = []
    for i, r in enumerate(routes):
        km  = r.get("total_km", 0)
        bh  = max(int(km / max_km * 120), 3)
        bx  = i * (bw + gap) + 8
        by  = ch - bh - 22
        col = _COLORS[i % len(_COLORS)]
        vid = r.get("vehicle_id", f"V{i+1}")
        bars.append(
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="{col}" rx="4" opacity=".9">'
            f'<title>{vid}: {km:.2f} km</title></rect>'
            f'<text x="{bx+bw//2}" y="{ch-5}" text-anchor="middle" '
            f'  font-size="10" fill="#666" font-family="sans-serif">{vid}</text>'
            f'<text x="{bx+bw//2}" y="{by-3}" text-anchor="middle" '
            f'  font-size="10" fill="{col}" font-weight="bold" font-family="sans-serif">{km:.1f}</text>')
    bar_w = len(routes) * (bw + gap) + 16
    bar_svg = f'<svg width="{bar_w}" height="{ch}" style="overflow:visible">{"".join(bars)}</svg>'

    # ââ Donut chart â cost breakdown âââââââââââââââââââââââââââââââââââââââââââ
    total_p = (travel_t + lateness_t) or 1
    cx2, cy2, ro, ri = 80, 80, 68, 40

    def _seg(sa, ea, col, tip):
        if ea - sa < 0.005:
            return ""
        x1 = cx2 + ro * math.cos(sa);  y1 = cy2 + ro * math.sin(sa)
        x2 = cx2 + ro * math.cos(ea);  y2 = cy2 + ro * math.sin(ea)
        xi = cx2 + ri * math.cos(ea);  yi = cy2 + ri * math.sin(ea)
        xj = cx2 + ri * math.cos(sa);  yj = cy2 + ri * math.sin(sa)
        lg = 1 if ea - sa > math.pi else 0
        return (f'<path d="M{x1:.1f},{y1:.1f} A{ro},{ro} 0 {lg} 1 {x2:.1f},{y2:.1f}'
                f' L{xi:.1f},{yi:.1f} A{ri},{ri} 0 {lg} 0 {xj:.1f},{yj:.1f} Z"'
                f' fill="{col}" opacity="0.92"><title>{tip}</title></path>')

    a0 = -math.pi / 2
    a1 = a0 + 2 * math.pi * travel_t / total_p
    a2 = a1 + 2 * math.pi * lateness_t / total_p
    donut = (
        _seg(a0, a1, "#3498db", f"Travel: {travel_t:.1f} min") +
        _seg(a1, a2, "#e74c3c", f"Lateness penalty: {lateness_t:.1f} min") +
        f'<text x="{cx2}" y="{cy2-5}" text-anchor="middle" '
        f'  font-size="10" fill="#888" font-family="sans-serif">Total</text>'
        f'<text x="{cx2}" y="{cy2+11}" text-anchor="middle" '
        f'  font-size="14" font-weight="bold" fill="#2c3e50" font-family="sans-serif">{obj:.0f}m</text>')
    donut_svg = f'<svg width="160" height="160">{donut}</svg>'

    # ââ Route details table ââââââââââââââââââââââââââââââââââââââââââââââââââââ
    trows = []
    for i, r in enumerate(routes):
        vid      = r.get("vehicle_id", "?")
        seq      = r.get("stop_sequence", [])
        n_stops  = max(len(seq) - 2, 0)
        load     = r.get("total_load", 0)
        km       = r.get("total_km", 0)
        cost_m   = r.get("estimated_cost_minutes", 0)
        col      = _COLORS[i % len(_COLORS)]
        stop_str = " &rarr; ".join(seq)
        trows.append(
            f'<tr>'
            f'<td><span style="display:inline-block;width:10px;height:10px;'
            f'background:{col};border-radius:50%;margin-right:4px;vertical-align:middle"></span>'
            f'<b>{vid}</b></td>'
            f'<td>{n_stops}</td><td>{load}</td>'
            f'<td>{km:.2f}</td><td>{cost_m:.1f}</td>'
            f'<td style="color:#888;font-size:11px;max-width:300px;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis">{stop_str}</td>'
            f'</tr>')

    # ââ Input summary ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    input_rows = []
    for c in customers[:15]:  # cap at 15 rows
        tw = c.get("time_window", "â")
        tw_str = f"{tw[0]}â{tw[1]}" if tw and tw != "â" else "â"
        input_rows.append(
            f'<tr><td>{c["id"]}</td>'
            f'<td>{float(c["lat"]):.4f}</td><td>{float(c["lon"]):.4f}</td>'
            f'<td>{c.get("demand", "â")}</td><td>{tw_str}</td>'
            f'<td>{c.get("service_time", 0)}</td></tr>')
    if len(customers) > 15:
        input_rows.append(
            f'<tr><td colspan="6" style="color:#888;font-style:italic">'
            f'â¦ {len(customers)-15} more customers</td></tr>')

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VRP Solution Dashboard</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', sans-serif; background: #eef2f7; color: #2c3e50; }}
header {{ background: #2c3e50; color: white; padding: 14px 20px;
          display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
header h2 {{ font-size: 16px; }}
.badge {{ padding: 3px 10px; border-radius: 12px; font-size: 11px;
          font-weight: bold; color: white; background: {status_c}; }}
.algo {{ margin-left: auto; font-size: 11px; opacity: .65; }}
.kpis {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 16px; }}
.kpi {{ background: white; border-radius: 10px; padding: 13px 18px;
        flex: 1 1 110px; box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; }}
.kpi .v {{ font-size: 22px; font-weight: 700; }}
.kpi .l {{ font-size: 11px; color: #95a5a6; margin-top: 2px; }}
.row {{ display: flex; flex-wrap: wrap; gap: 14px; padding: 0 16px 16px; }}
.card {{ background: white; border-radius: 10px; padding: 15px;
         box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
.card h3 {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase;
            letter-spacing: .5px; margin-bottom: 12px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th {{ background: #f5f7fa; padding: 7px 10px; text-align: left;
      font-weight: 600; font-size: 11px; color: #555; }}
td {{ padding: 6px 10px; border-bottom: 1px solid #f0f2f5; }}
tr:last-child td {{ border: none; }}
tr:hover td {{ background: #fafbfc; }}
.legend-dot {{ display: inline-block; width: 10px; height: 10px;
               border-radius: 50%; }}
.section-nav {{ display: flex; gap: 0; padding: 0 16px 8px; }}
.tab {{ padding: 8px 18px; cursor: pointer; background: #dde3ea; font-size: 13px;
        border: none; color: #555; }}
.tab.active {{ background: white; color: #2c3e50; font-weight: 600;
               border-radius: 6px 6px 0 0; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
</style>
<script>
function showTab(id) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  document.getElementById('content-' + id).classList.add('active');
}}
</script>
</head><body>
<header>
  <h2>&#x1F4CA; VRP Solution Dashboard</h2>
  <span class="badge">{status.upper()}</span>
  <span class="algo">{algo}</span>
</header>

<div class="kpis">{kpis_html}</div>

<div class="section-nav">
  <button class="tab active" id="tab-results" onclick="showTab('results')">Solution</button>
  <button class="tab" id="tab-charts" onclick="showTab('charts')">Charts</button>
  <button class="tab" id="tab-input" onclick="showTab('input')">Input Data</button>
</div>

<div class="tab-content active" id="content-results">
  <div class="row" style="padding-top:0">
    <div class="card" style="flex:1;min-width:320px">
      <h3>Route Details</h3>
      <table>
        <thead><tr>
          <th>Vehicle</th><th>Stops</th><th>Load</th>
          <th>km</th><th>Min</th><th>Sequence</th>
        </tr></thead>
        <tbody>{"".join(trows)}</tbody>
      </table>
    </div>
  </div>
</div>

<div class="tab-content" id="content-charts">
  <div class="row" style="padding-top:0">
    <div class="card">
      <h3>Distance per Vehicle (km)</h3>
      {bar_svg}
    </div>
    <div class="card">
      <h3>Cost Breakdown</h3>
      {donut_svg}
      <div style="font-size:11px;margin-top:6px">
        <span class="legend-dot" style="background:#3498db"></span>&nbsp;Travel {travel_t:.1f} min
        &emsp;
        <span class="legend-dot" style="background:#e74c3c"></span>&nbsp;Lateness {lateness_t:.1f} min
      </div>
    </div>
  </div>
</div>

<div class="tab-content" id="content-input">
  <div class="row" style="padding-top:0">
    <div class="card" style="flex:1">
      <h3>Customer Locations</h3>
      <table>
        <thead><tr>
          <th>ID</th><th>Lat</th><th>Lon</th>
          <th>Demand</th><th>Time Window</th><th>Service (min)</th>
        </tr></thead>
        <tbody>{"".join(input_rows)}</tbody>
      </table>
    </div>
  </div>
</div>

</body></html>"""
