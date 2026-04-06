"""
OceanMaster v13.2 — Captain-Friendly Dashboard
=================================================
Key features:
- Ultra-dark map with prominent ocean currents + EEZ boundaries
- Fish species NAME labels on every hotspot marker
- Large-font AI analysis popup on click (why here, what data)
- Typhoon simulation with predicted path + evasion advice
- Clear navigation route lines from vessel to hotspots
"""


def get_css():
    return '''
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Noto Sans TC',sans-serif;background:#010812;color:#c0d0e0;overflow:hidden;height:100vh}
.dashboard{display:flex;flex-direction:column;height:100vh}

/* TOP BAR */
.top-bar{height:42px;background:#020c18;border-bottom:1px solid #0a2040;display:flex;align-items:center;padding:0 16px;gap:12px;z-index:100}
.top-bar .logo{font-family:'Rajdhani',sans-serif;font-size:16px;font-weight:700;color:#0080bb;white-space:nowrap}
.vessel-pos{font-family:'Share Tech Mono',monospace;font-size:12px;color:#00aa88;background:rgba(0,170,136,0.06);padding:2px 10px;border-radius:3px;border:1px solid rgba(0,170,136,0.1)}
.chip{font-size:10px;padding:2px 8px;border-radius:10px;border:1px solid rgba(255,255,255,0.05)}
.chip.safe{background:rgba(40,160,80,0.08);color:#28a050}.chip.caution{background:rgba(200,160,30,0.08);color:#c8a01e}
.chip.danger{background:rgba(200,50,60,0.1);color:#c8323c}
.typhoon-flash{animation:tflash 1s ease-in-out infinite}
@keyframes tflash{0%,100%{opacity:1}50%{opacity:0.3}}
.spacer{flex:1}
.clock{font-family:'Share Tech Mono',monospace;font-size:12px;color:#2a4a6a}

/* MAIN */
.main{display:flex;flex:1;overflow:hidden}

/* LEFT */
.left{width:260px;min-width:260px;background:#030d1a;border-right:1px solid #0a2040;display:flex;flex-direction:column}
.left-hd{padding:8px 10px;border-bottom:1px solid #0a2040;font-family:'Rajdhani',sans-serif;font-size:13px;font-weight:700;color:#0080bb}
.cards{flex:1;overflow-y:auto;padding:4px}
.cards::-webkit-scrollbar{width:3px}.cards::-webkit-scrollbar-thumb{background:#0a2040}
.hcard{background:#041020;border-radius:5px;padding:7px 8px;margin-bottom:3px;cursor:pointer;transition:all 0.15s;border-left:3px solid transparent}
.hcard:hover{background:#081830}.hcard.active{background:#081830;box-shadow:0 0 8px rgba(0,128,187,0.15)}
.hcard.t-high{border-left-color:#c83040}.hcard.t-mid{border-left-color:#b89520}.hcard.t-low{border-left-color:#28a050}
.hcard .top{display:flex;align-items:center;gap:5px}
.hcard .rk{font-family:'Rajdhani';font-size:16px;font-weight:700;color:#0080bb}
.hcard .sp{font-size:11px;color:#a0b0c0;flex:1}
.hcard .hsi{font-family:'Rajdhani';font-size:12px;font-weight:600}
.hcard .hsi.high{color:#c83040}.hcard .hsi.mid{color:#b89520}.hcard .hsi.low{color:#28a050}
.hcard .sub{font-family:'Share Tech Mono';font-size:7px;color:#2a4a6a;margin-top:1px}
.hcard .tags{display:flex;gap:2px;margin-top:3px;flex-wrap:wrap}
.htag{font-size:6px;padding:0 3px;border-radius:2px;background:rgba(0,128,187,0.06);color:#0080bb;border:1px solid rgba(0,128,187,0.08)}
.left-ft{padding:5px 10px;border-top:1px solid #0a2040;display:flex;justify-content:space-around;font-size:7px;color:#2a4a6a}
.left-ft .n{font-family:'Rajdhani';font-size:13px;font-weight:700;color:#00aa88}

/* MAP */
.center{flex:1;position:relative;background:#010810}
#map{width:100%;height:100%}
@keyframes pulse{0%{transform:scale(1);opacity:1}50%{transform:scale(1.4);opacity:0.15}100%{transform:scale(1);opacity:0}}

.mctrls{position:absolute;top:8px;right:8px;z-index:1000;display:flex;flex-direction:column;gap:2px}
.cbtn{background:rgba(2,12,24,0.93);border:1px solid #0a2040;color:#3a5a7a;padding:3px 7px;border-radius:3px;font-size:8px;cursor:pointer;font-family:'Noto Sans TC'}
.cbtn:hover{color:#6a8aa0}.cbtn.on{border-color:#0060a0;color:#0080bb;background:rgba(0,128,187,0.06)}

.mleg{position:absolute;bottom:10px;left:10px;z-index:1000;background:rgba(2,12,24,0.93);padding:6px 10px;border-radius:5px;border:1px solid #0a2040;font-size:8px}
.mleg .t{font-family:'Rajdhani';font-size:9px;font-weight:600;color:#0080bb;margin-bottom:2px}
.mleg .r{display:flex;align-items:center;gap:3px;margin-bottom:1px}
.ldot{border-radius:50%;display:inline-block}

/* RIGHT */
.right{width:380px;min-width:380px;background:#030d1a;border-left:1px solid #0a2040;display:flex;flex-direction:column}
.tabs{display:flex;border-bottom:1px solid #0a2040;background:#020a16}
.tbtn{flex:1;padding:6px 2px;font-size:9px;color:#2a4a6a;cursor:pointer;border:none;background:none;text-align:center;border-bottom:2px solid transparent;font-family:'Noto Sans TC'}
.tbtn:hover{color:#5a7a9a}.tbtn.active{color:#0080bb;border-bottom-color:#0060a0}
.tpane{flex:1;overflow-y:auto;padding:10px 14px;display:none}
.tpane.active{display:block}
.tpane::-webkit-scrollbar{width:3px}.tpane::-webkit-scrollbar-thumb{background:#0a2040}

.stitle{font-family:'Rajdhani';font-size:13px;font-weight:600;color:#0080bb;letter-spacing:0.5px;margin:14px 0 6px;padding-bottom:3px;border-bottom:1px solid rgba(0,128,187,0.1)}
.stitle:first-child{margin-top:0}

/* AI Analysis Box - BIG & CLEAR for captain */
.ai-box{background:#041020;border-radius:6px;padding:12px;margin:8px 0;border:1px solid rgba(0,128,187,0.15);
  font-size:14px;color:#a0c0d0;line-height:1.8;white-space:pre-line}
.ai-box b{color:#0080bb}
.ai-factor{background:#041020;border-radius:5px;padding:8px 10px;margin:4px 0;border-left:3px solid #0060a0;font-size:13px;color:#90a8b8}
.ai-factor .fname{color:#0080bb;font-weight:600;font-size:14px}
.ai-factor .fval{font-family:'Rajdhani';font-size:18px;font-weight:700;color:#00aa88;float:right}

.score-row{display:flex;align-items:center;margin-bottom:5px}
.score-lbl{width:80px;font-size:11px;color:#5a7a9a}
.score-bg{flex:1;height:6px;background:#081828;border-radius:3px;overflow:hidden}
.score-fill{height:100%;border-radius:3px}
.score-v{width:50px;text-align:right;font-family:'Rajdhani';font-size:13px;font-weight:600;color:#a0b0c0;margin-left:6px}

.dgrid{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-bottom:10px}
.dcell{background:#041020;border-radius:4px;padding:6px 8px;border:1px solid rgba(10,32,64,0.4)}
.dcell .dl{font-size:8px;color:#2a4a6a;text-transform:uppercase}
.dcell .dv{font-family:'Rajdhani';font-size:18px;font-weight:700;color:#00aa88;line-height:1.1}
.dcell .du{font-size:9px;color:#2a4a6a}
.dcell .ds{font-size:6px;margin-top:1px;padding:0 3px;border-radius:1px;display:inline-block}
.dcell .ds.live{background:rgba(40,160,80,0.08);color:#28a050}.dcell .ds.hist{background:rgba(0,80,140,0.06);color:#2a5a8a}

.alert{padding:8px 10px;border-radius:5px;margin-bottom:6px;font-size:12px;line-height:1.4}
.alert.red{background:rgba(200,48,64,0.06);border:1px solid rgba(200,48,64,0.12);color:#c06070}
.alert.yellow{background:rgba(184,149,32,0.06);border:1px solid rgba(184,149,32,0.1);color:#b89520}
.alert.blue{background:rgba(0,80,140,0.05);border:1px solid rgba(0,80,140,0.08);color:#4a80aa}

.route-grid{background:#041020;border-radius:5px;padding:10px;margin:6px 0;border:1px solid rgba(10,32,64,0.3);display:grid;grid-template-columns:1fr 1fr;gap:8px}
.route-grid .rv{font-family:'Rajdhani';font-size:20px;font-weight:700;color:#b89520;text-align:center}
.route-grid .rl{font-size:9px;color:#2a4a6a;text-align:center}

.sp-row{display:flex;align-items:center;gap:4px;margin-bottom:3px;font-size:11px}
.sp-row .sn{width:55px;color:#5a7a9a}.sp-row .sb{flex:1;height:6px;background:#081828;border-radius:3px;overflow:hidden}
.sp-row .sf{height:100%;border-radius:3px}.sp-row .sv{width:30px;text-align:right;font-family:'Rajdhani';font-weight:600;color:#a0b0c0}

.shap-row{display:flex;align-items:center;margin-bottom:3px;font-size:9px}
.shap-row .sl{width:70px;color:#5a7a9a}.shap-row .sb{flex:1;height:7px;background:#081828;border-radius:2px;overflow:hidden;margin:0 4px}
.shap-row .sf{height:100%;border-radius:2px}
.shap-row .sf.pos{background:rgba(0,170,136,0.5)}.shap-row .sf.neg{background:rgba(200,48,64,0.4)}

.leaflet-popup-content-wrapper{background:rgba(3,13,26,0.96)!important;color:#c0d0e0!important;border:1px solid #0a2040!important;border-radius:6px!important}
.leaflet-popup-tip{background:rgba(3,13,26,0.96)!important}

.typhoon-spin{animation:tspin 3s linear infinite}
@keyframes tspin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}

.hook-d{display:flex;gap:2px;height:20px;margin:4px 0}
.hook-d div{flex:1;border-radius:2px;display:flex;align-items:center;justify-content:center;font-size:8px;font-family:'Share Tech Mono';color:rgba(200,210,220,0.3)}
.hook-d .opt{background:rgba(0,80,140,0.1);border:1px solid #0060a0;color:#00aa88;font-weight:600;font-size:9px}
.hook-d .sh{background:rgba(40,160,80,0.05);border:1px solid rgba(40,160,80,0.1)}
.hook-d .dp{background:rgba(100,60,130,0.05);border:1px solid rgba(100,60,130,0.1)}

.placeholder{text-align:center;color:#1a3a5a;margin-top:50px;font-size:12px}
'''


def get_html_structure(ts_display, vessel_lat, vessel_lon, home_port_name,
                       n_total, n_safe, n_caution, n_avoid, lunar_str, enso_str, has_typhoon):
    sc = "danger" if n_avoid > 0 else ("caution" if n_caution > 3 else "safe")
    st = f"🔴 {n_avoid}危險" if n_avoid > 0 else (f"🟡 {n_caution}注意" if n_caution > 3 else "🟢 安全")
    ty = '<span class="chip typhoon-flash" style="background:rgba(200,50,60,0.1);color:#e04050;border-color:rgba(200,50,60,0.2)">🌀 颱風警報</span>' if has_typhoon else '<span class="chip safe">🌀 無颱風</span>'
    return f'''
<div class="dashboard">
<div class="top-bar">
  <div class="logo">🚢 OceanMaster Captain v16</div>
  <div class="vessel-pos">⚓ {vessel_lat:.4f}°N {vessel_lon:.4f}°E</div>
  <span class="chip {sc}">{st}</span>{ty}
  <span class="chip" style="color:#2a5a8a">🌊 {enso_str}</span>
  <span class="chip" style="color:#5a5020">{lunar_str}</span>
  <div class="spacer"></div>
  <div class="clock">{ts_display} UTC+8</div>
</div>
<div class="main">
<div class="left">
  <div class="left-hd">📡 漁場熱點 ({n_total})</div>
  <div class="cards" id="CL"></div>
  <div class="left-ft">
    <div><div class="n">{n_total}</div>熱點</div>
    <div><div class="n" style="color:#28a050">{n_safe}</div>安全</div>
    <div><div class="n" style="color:#b89520">{n_caution}</div>注意</div>
    <div><div class="n" style="color:#c83040">{n_avoid}</div>危險</div>
  </div>
</div>
<div class="center">
  <div id="map"></div>
  <div class="mctrls">
    <button class="cbtn on" id="bEez" onclick="tgl('eez')">🗺 EEZ漁區</button>
    <button class="cbtn on" id="bRoute" onclick="tgl('route')">🧭 航線</button>
    <button class="cbtn on" id="bCurr" onclick="tgl('curr')">🌊 海流</button>
    <button class="cbtn on" id="bGrid" onclick="tgl('grid')">📐 經緯度</button>
    <button class="cbtn on" id="bTyph" onclick="tgl('typh')">🌀 颱風</button>
    <button class="cbtn" id="bGfw" onclick="tgl('gfw')">🛰 AIS船舶</button>
  </div>
  <div class="mleg">
    <div class="t">圖例</div>
    <div class="r"><span class="ldot" style="width:7px;height:7px;background:#c83040"></span>高潛力</div>
    <div class="r"><span class="ldot" style="width:5px;height:5px;background:#b89520"></span>中潛力</div>
    <div class="r"><span class="ldot" style="width:4px;height:4px;background:#28a050"></span>低潛力</div>
    <div class="r" style="margin-top:2px;border-top:1px solid rgba(255,255,255,0.03);padding-top:2px">🚢本船 → 海流 --- EEZ 🌀颱風</div>
  </div>
</div>
<div class="right">
  <div class="tabs">
    <button class="tbtn active" onclick="stab('det')">📊 AI分析</button>
    <button class="tbtn" onclick="stab('nav')">🧭 航行</button>
    <button class="tbtn" onclick="stab('saf')">⚠️ 安全</button>
    <button class="tbtn" onclick="stab('ocn')">🌊 環境</button>
  </div>
  <div class="tpane active" id="tab-det"><div class="placeholder">🎯<br>點擊地圖上的魚群熱點<br>查看 AI 分析報告</div></div>
  <div class="tpane" id="tab-nav"></div>
  <div class="tpane" id="tab-saf"></div>
  <div class="tpane" id="tab-ocn"></div>
</div>
</div></div>
'''


def get_javascript(records_json, gfw_pts_js, center_lat, center_lon,
                   vessel_lat, vessel_lon, home_port_name, eez_js):
    sp_colors = '{"skipjack":"#28a050","yellowfin":"#b89520","bigeye":"#2868a0","albacore":"#c83040","japanese_flying_squid":"#7040a0","pacific_saury":"#188880","mahi_mahi":"#b86020","blue_marlin":"#2040a0"}'
    sp_zh = '{"skipjack":"鰹魚","yellowfin":"黃鰭鮪","bigeye":"大目鮪","albacore":"長鰭鮪","japanese_flying_squid":"日本魷","pacific_saury":"秋刀魚","mahi_mahi":"鬼頭刀","blue_marlin":"旗魚"}'

    return f'''
const D={records_json};
const gD={gfw_pts_js};
const SC={sp_colors};
const SN={sp_zh};
const VP=[{vessel_lat},{vessel_lon}];

// === DARK MAP (original style) ===
const map=L.map('map',{{zoomControl:true,attributionControl:false}}).setView([{center_lat:.2f},{center_lon:.2f}],5);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{maxZoom:19,subdomains:'abcd'}}).addTo(map);

// Vessel
L.marker(VP,{{icon:L.divIcon({{html:'<div style="font-size:26px;filter:drop-shadow(0 0 8px rgba(0,128,187,0.6))">🚢</div>',className:'',iconSize:[30,30],iconAnchor:[15,15]}}),zIndexOffset:2000}}).addTo(map)
 .bindPopup('<div style="font-size:14px"><b style="color:#0080bb">🚢 {home_port_name}</b><br><span style="color:#00aa88;font-family:Share Tech Mono">'+VP[0].toFixed(4)+'°N '+VP[1].toFixed(4)+'°E</span></div>');

// === LAYERS ===
const L_=  {{}};
const L_on={{eez:1,route:1,curr:1,grid:1,typh:1,gfw:0}};

// --- EEZ BOUNDARIES (thick, bright, clear) ---
L_.eez=L.layerGroup();
{eez_js}
L_.eez.addTo(map);

// --- NAVIGATION ROUTES (from vessel to each hotspot) ---
L_.route=L.layerGroup();
D.forEach(d=>{{
  const c=d.safety_level==='AVOID'?'#c83040':d.safety_level==='CAUTION'?'#b89520':'#0060a0';
  const line=L.polyline([VP,[d.lat,d.lon]],{{color:c,weight:2,opacity:0.5,dashArray:'8,6'}});
  line.bindTooltip(`${{d.species_zh}} #${{d.rank}} — ${{d.distance_km}}km · ${{d.transit_days}}天`,{{permanent:false,className:'',direction:'center'}});
  line.addTo(L_.route);
  // 方位標籤
  const mid=[(VP[0]+d.lat)/2,(VP[1]+d.lon)/2];
  L.marker(mid,{{icon:L.divIcon({{html:`<span style="font-size:8px;color:${{c}};opacity:0.6;font-family:Share Tech Mono">${{d.bearing}} ${{d.distance_km}}km</span>`,className:'',iconAnchor:[15,5]}})}} ).addTo(L_.route);
}});
L_.route.addTo(map);

// --- OCEAN CURRENTS (prominent arrows) ---
L_.curr=L.layerGroup();
const currents=[
  [22,119,25,0.4,'臺灣海峽流',1.5],[24,121,35,0.8,'黑潮(臺灣)',2.5],[26,123,40,1.2,'黑潮主流',3],
  [28,126,45,1.0,'黑潮',2.8],[30,130,50,0.8,'黑潮延伸',2.5],[32,136,55,0.6,'黑潮延伸',2],
  [34,142,60,0.5,'黑潮→北太平洋',1.8],[36,148,70,0.4,'北太平洋流',1.5],
  [20,128,280,0.5,'北赤道流',2],[18,132,275,0.5,'北赤道流',2],[15,138,285,0.4,'北赤道流',1.5],
  [10,130,310,0.3,'南赤道流',1.2],[35,143,190,0.5,'親潮(冷)',2],[33,145,200,0.4,'親潮',1.5],
  [12,122,350,0.3,'呂宋海流',1.2],[25,125,40,0.9,'黑潮支流',2.5]
];
currents.forEach(([lat,lon,deg,spd,name,w])=>{{
  const r=deg*Math.PI/180, d1=1.2;
  const lat2=lat+d1*Math.cos(r), lon2=lon+d1*Math.sin(r);
  const op=Math.min(0.7, spd*0.5+0.2);
  // Main arrow line
  L.polyline([[lat,lon],[lat2,lon2]],{{color:'#1070b0',weight:w,opacity:op}}).addTo(L_.curr);
  // Arrowhead
  const al=0.35,a1=(deg+155)*Math.PI/180,a2=(deg-155)*Math.PI/180;
  L.polyline([[lat2,lon2],[lat2+al*Math.cos(a1),lon2+al*Math.sin(a1)]],{{color:'#1070b0',weight:w*0.7,opacity:op}}).addTo(L_.curr);
  L.polyline([[lat2,lon2],[lat2+al*Math.cos(a2),lon2+al*Math.sin(a2)]],{{color:'#1070b0',weight:w*0.7,opacity:op}}).addTo(L_.curr);
  // Label
  L.marker([(lat+lat2)/2,(lon+lon2)/2],{{icon:L.divIcon({{html:`<span style="font-size:8px;color:#1070b0;opacity:0.7;font-weight:600">${{name}} ${{spd}}m/s</span>`,className:'',iconAnchor:[-3,-3]}})}} ).addTo(L_.curr);
}});
L_.curr.addTo(map);

// --- GRATICULE ---
L_.grid=L.layerGroup();
for(let la=0;la<=45;la+=5){{
  L.polyline([[la,108],[la,180]],{{color:'rgba(30,60,90,0.15)',weight:0.4,dashArray:'2,6'}}).addTo(L_.grid);
  L.marker([la,112],{{icon:L.divIcon({{html:`<span style="font-size:8px;color:#15304a">${{la}}°N</span>`,className:'',iconAnchor:[0,4]}})}} ).addTo(L_.grid);
}}
for(let lo=110;lo<=175;lo+=5){{
  L.polyline([[-5,lo],[45,lo]],{{color:'rgba(30,60,90,0.15)',weight:0.4,dashArray:'2,6'}}).addTo(L_.grid);
  L.marker([0.5,lo],{{icon:L.divIcon({{html:`<span style="font-size:8px;color:#15304a">${{lo}}°E</span>`,className:'',iconAnchor:[8,0]}})}} ).addTo(L_.grid);
}}
L_.grid.addTo(map);

// --- TYPHOON SIMULATION ---
L_.typh=L.layerGroup();
const tyPath=[[7,148],[9,145],[11,142],[13,139],[15,136],[17,133],[19,130],[21,127],[23,125]];
// Historical path (solid)
L.polyline(tyPath.slice(0,4),{{color:'#c83040',weight:2.5,opacity:0.6}}).addTo(L_.typh);
// Current position
L.marker(tyPath[3],{{icon:L.divIcon({{html:'<div style="font-size:28px" class="typhoon-spin">🌀</div>',className:'',iconSize:[32,32],iconAnchor:[16,16]}}),zIndexOffset:1500}}).addTo(L_.typh);
L.marker(tyPath[3],{{icon:L.divIcon({{html:'<div style="font-size:11px;color:#c83040;font-weight:700;text-shadow:0 0 4px rgba(0,0,0,0.8)">TD-03 熱帶低壓<br><span style="font-size:9px;color:#b07070">風速 15m/s · 氣壓 998hPa</span></div>',className:'',iconAnchor:[-20,10]}})}} ).addTo(L_.typh);
// Impact circles
L.circle(tyPath[3],{{radius:150000,color:'#c83040',fillColor:'#c83040',fillOpacity:0.03,weight:1,dashArray:'4,4'}}).addTo(L_.typh);
L.circle(tyPath[3],{{radius:350000,color:'#b89520',fillColor:'#b89520',fillOpacity:0.015,weight:0.6,dashArray:'4,4'}}).addTo(L_.typh);
L.marker([tyPath[3][0]+1.2,tyPath[3][1]-0.5],{{icon:L.divIcon({{html:'<span style="font-size:7px;color:#c83040">暴風圈 150km</span>',className:''}})}} ).addTo(L_.typh);
L.marker([tyPath[3][0]+3,tyPath[3][1]-1],{{icon:L.divIcon({{html:'<span style="font-size:7px;color:#b89520">警戒範圍 350km</span>',className:''}})}} ).addTo(L_.typh);
// Predicted path (dashed)
L.polyline(tyPath.slice(3),{{color:'#c83040',weight:2,opacity:0.4,dashArray:'6,8'}}).addTo(L_.typh);
// Future markers
tyPath.slice(4).forEach((p,i)=>{{
  L.circleMarker(p,{{radius:3+i*0.8,color:'#c83040',fillColor:'#c83040',fillOpacity:0.15,weight:0.6}}).addTo(L_.typh);
}});
['','','','','+12h','+24h','+36h','+48h','+60h'].forEach((t,i)=>{{
  if(t&&tyPath[i])L.marker(tyPath[i],{{icon:L.divIcon({{html:`<span style="font-size:8px;color:#905050;font-weight:600">${{t}}</span>`,className:'',iconAnchor:[-8,5]}})}} ).addTo(L_.typh);
}});
// Evasion suggestion line
L.polyline([VP,[VP[0]-2,VP[1]+5],[VP[0]-3,VP[1]+10]],{{color:'#28a050',weight:1.5,opacity:0.4,dashArray:'3,5'}}).addTo(L_.typh);
L.marker([VP[0]-2.5,VP[1]+7],{{icon:L.divIcon({{html:'<span style="font-size:8px;color:#28a050">✅ 建議規避路線</span>',className:'',iconAnchor:[30,5]}})}} ).addTo(L_.typh);
L_.typh.addTo(map);

// GFW
if(gD.length>0){{L_.gfw=L.heatLayer(gD,{{radius:15,blur:18,maxZoom:10,max:1.0,gradient:{{0:'transparent',0.3:'rgba(0,0,80,0.15)',0.6:'rgba(0,60,120,0.2)',0.8:'rgba(150,120,20,0.3)',1:'rgba(180,40,40,0.4)'}}}})}}

function tgl(n){{
  const b=document.getElementById('b'+n.charAt(0).toUpperCase()+n.slice(1));
  if(!L_[n])return;
  L_on[n]=!L_on[n];
  if(L_on[n]){{L_[n].addTo(map);if(b)b.classList.add('on')}}
  else{{map.removeLayer(L_[n]);if(b)b.classList.remove('on')}}
}}

// [v15.3] 點擊地圖空白處顯示經緯度座標
map.on('click',function(e){{
  const la=e.latlng.lat,lo=e.latlng.lng;
  const dLat=la-VP[0],dLon=lo-VP[1];
  const R=3440.065,rLat=dLat*Math.PI/180,rLon=dLon*Math.PI/180;
  const a=Math.sin(rLat/2)**2+Math.cos(VP[0]*Math.PI/180)*Math.cos(la*Math.PI/180)*Math.sin(rLon/2)**2;
  const dist=(2*R*Math.asin(Math.sqrt(a))).toFixed(1);
  const brg=((Math.atan2(Math.sin(rLon)*Math.cos(la*Math.PI/180),Math.cos(VP[0]*Math.PI/180)*Math.sin(la*Math.PI/180)-Math.sin(VP[0]*Math.PI/180)*Math.cos(la*Math.PI/180)*Math.cos(rLon))*180/Math.PI+360)%360).toFixed(0);
  const dirs=['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
  const dirStr=dirs[Math.round(brg/22.5)%16];
  L.popup({{className:'',maxWidth:260}})
   .setLatLng(e.latlng)
   .setContent(`<div style="font-size:14px;font-family:'Noto Sans TC',sans-serif">` +
     `<div style="font-size:16px;font-weight:700;color:#0080bb;margin-bottom:4px">📍 位置座標</div>` +
     `<div style="font-family:'Share Tech Mono',monospace;font-size:15px;color:#00aa88;margin-bottom:6px">${{la.toFixed(4)}}°N ${{lo.toFixed(4)}}°E</div>` +
     `<div style="color:#5a7a9a;font-size:12px">🚢 距本船: <b style="color:#b89520">${{dist}} nm</b> · 方位 <b style="color:#b89520">${{dirStr}} (${{brg}}°)</b></div>` +
     `</div>`)
   .openOn(map);
}});

// === HOTSPOT MARKERS WITH FISH NAME LABELS ===
const M={{}};
D.forEach(d=>{{
  const sz=d.marker_radius*2;
  const sc=d.safety_level==='AVOID'?'#c83040':d.safety_level==='CAUTION'?'#b89520':'#28a050';
  // Marker with fish name label
  const html=`<div style="position:relative;text-align:center">` +
    `<div style="width:${{sz}}px;height:${{sz}}px;border-radius:50%;background:${{d.marker_color}};opacity:0.75;border:1.5px solid rgba(255,255,255,0.1);margin:0 auto"></div>` +
    `<div style="position:absolute;top:-3px;right:-5px;width:8px;height:8px;border-radius:50%;background:${{sc}};border:1px solid #010812"></div>` +
    `<div style="margin-top:2px;font-size:10px;font-weight:600;color:${{d.marker_color}};text-shadow:0 0 6px rgba(0,0,0,0.9),0 0 2px rgba(0,0,0,1);white-space:nowrap;font-family:'Noto Sans TC'">#${{d.rank}} ${{d.species_zh}}</div>` +
    `</div>`;
  const m=L.marker([d.lat,d.lon],{{icon:L.divIcon({{html,className:'',iconSize:[sz+30,sz+20],iconAnchor:[(sz+30)/2,sz/2]}})}}).addTo(map);
  // Click marker → directly show AI analysis in right panel (no popup blocking)
  m.on('click',()=>{{map.closePopup();show(d.id)}});
  M[d.id]=m;
}});

// === LEFT CARDS ===
const cl=document.getElementById('CL');
D.forEach(d=>{{
  const c=document.createElement('div');
  c.className=`hcard t-${{d.tier}}`;c.id=`c-${{d.id}}`;c.onclick=()=>show(d.id);
  const dc=d.safety_level==='AVOID'?'#c83040':d.safety_level==='CAUTION'?'#b89520':'#28a050';
  let tg=`<span class="htag" style="color:${{dc}}">${{d.safety_emoji}}</span>`;
  if(d.route_cost_usd>0)tg+=`<span class="htag">💰$${{Math.round(d.route_cost_usd/1000)}}K</span>`;
  if(d.shear_risk_level>=2)tg+=`<span class="htag" style="color:#c05060">⚠剪切</span>`;
  tg+=`<span class="htag">${{d.bearing}} ${{d.transit_days}}天</span>`;
  const tv=d.travel_days>0?d.travel_days+'天'+d.travel_hours+'時':d.travel_hours+'時';
  c.innerHTML=`<div class="top"><span class="rk">#${{d.rank}}</span><span class="sp"><span class="safety-dot" style="background:${{dc}};width:5px;height:5px;border-radius:50%;display:inline-block"></span> ${{d.species_zh}}</span><span class="hsi ${{d.tier}}">HSI ${{d.hsi_str}}</span></div><div class="sub">${{d.lat}}°N ${{d.lon}}°E · ${{tv}} · ${{d.eez}}</div><div class="tags">${{tg}}</div>`;
  cl.appendChild(c);
}});

// === TABS ===
function stab(n){{document.querySelectorAll('.tbtn').forEach(b=>b.classList.remove('active'));document.querySelectorAll('.tpane').forEach(t=>t.classList.remove('active'));event.target.classList.add('active');document.getElementById('tab-'+n).classList.add('active')}}

// === SHOW DETAIL (AI Analysis) ===
function show(id){{
  const d=D.find(x=>x.id===id);if(!d)return;
  document.querySelectorAll('.hcard').forEach(c=>c.classList.remove('active'));
  const card=document.getElementById('c-'+id);
  if(card){{card.classList.add('active');card.scrollIntoView({{behavior:'smooth',block:'nearest'}})}}
  map.flyTo([d.lat,d.lon],7,{{duration:0.6}});
  // Switch to detail tab
  document.querySelectorAll('.tbtn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tpane').forEach(t=>t.classList.remove('active'));
  document.querySelector('.tbtn').classList.add('active');
  document.getElementById('tab-det').classList.add('active');

  const sc=d.safety_level==='AVOID'?'#c83040':d.safety_level==='CAUTION'?'#b89520':'#28a050';
  const fc=(v,u,c,l)=>v!==null&&v!==0?`<span class="dv">${{v}}</span><span class="du">${{u}}</span><div class="ds ${{c}}">${{l}}</div>`:`<span class="dv" style="color:#0a2040">—</span>`;
  const df=d.data_freshness||{{}};
  const sm=k=>{{const v=df[k]||'';return v==='NRT'?['live','即時']:v==='歷史均值'?['hist','歷史']:['hist','估算']}};

  // === SHAP bars ===
  const sLbl={{sst:'🌡 海水溫度',chl:'🌿 葉綠素',ssh:'🌊 海面高度',do:'💧 溶氧量',current_speed:'🌊 海流速度',front_strength:'🔥 鋒面強度',eddy_strength:'🌀 渦旋強度',phi:'📊 Φ商業指數',bathy_depth:'⛰ 水深',dist_seamount:'🏔 海山距離',bathy_slope:'📐 海底坡度',dist_shelf_break:'🏖 陸棚距離'}};
  let shap='';
  if(d.shap_values){{
    const e=Object.entries(d.shap_values).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,8);
    const mx=Math.max(...e.map(x=>Math.abs(x[1])),0.01);
    e.forEach(([k,v])=>{{const p=Math.min(Math.abs(v)/mx*100,100);
      shap+=`<div class="shap-row"><span class="sl">${{sLbl[k]||k}}</span><div class="sb"><div class="sf ${{v>=0?'pos':'neg'}}" style="width:${{p}}%"></div></div><span style="font-size:8px;color:#3a5a7a;width:28px">${{v>0?'+':''}}${{v.toFixed(1)}}</span></div>`}})
  }}

  // === Species probs ===
  let sp='';
  if(d.species_probs){{Object.entries(d.species_probs).sort((a,b)=>b[1]-a[1]).forEach(([k,v])=>{{
    const p=Math.round(v*100);
    sp+=`<div class="sp-row"><span class="sn">${{SN[k]||k}}</span><div class="sb"><div class="sf" style="width:${{p}}%;background:${{SC[k]||'#555'}}"></div></div><span class="sv">${{p}}%</span></div>`}})}}

  // === AI Analysis text (BIG FONT for captain) ===
  const aiText = d.explain || '暫無 AI 分析資料';

  // === Key factors display ===
  let factors='';
  if(d.shap_values){{
    const top3=Object.entries(d.shap_values).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,4);
    top3.forEach(([k,v])=>{{
      const icon=v>=0?'📈':'📉';
      const dataVal = k==='sst'?d.sst+'°C':k==='chl'?d.chl+'mg/m³':k==='bathy_depth'?d.depth_m+'m':k==='front_strength'?d.chl_front_strength:k==='eddy_strength'?d.eddy_type:k==='do'?d.do_surface+'μmol':k==='phi'?d.phi:'—';
      factors+=`<div class="ai-factor"><span class="fname">${{icon}} ${{sLbl[k]||k}}</span><span class="fval">${{dataVal}}</span><div style="font-size:11px;color:#5a7a9a;margin-top:2px">SHAP 貢獻: ${{v>0?'+':''}}${{v.toFixed(2)}} (${{v>=0?'正向加分':'負向減分'}})</div></div>`}})
  }}

  document.getElementById('tab-det').innerHTML=`
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      <span style="font-family:Rajdhani;font-size:28px;font-weight:700;color:${{d.marker_color}}">#${{d.rank}}</span>
      <div style="flex:1"><div style="font-size:16px;font-weight:600;color:#b0c0d0">${{d.species_zh}}</div><div style="font-family:'Share Tech Mono';font-size:10px;color:#2a4a6a">${{d.lat}}°N ${{d.lon}}°E · ${{d.eez}}</div></div>
      <span style="background:${{sc}}10;color:${{sc}};padding:3px 8px;border-radius:4px;font-size:11px;border:1px solid ${{sc}}30">${{d.safety_emoji}} ${{d.safety_level}}</span>
    </div>
    ${{d.shear_warning?`<div class="alert red">${{d.shear_warning}}</div>`:''}}
    ${{d.safety_level!=='SAFE'?`<div class="alert yellow">${{d.safety_emoji}} ${{d.safety_reason}}</div>`:''}}

    <div class="stitle">🤖 AI 漁場分析 — 為何推薦此位置</div>
    <div class="ai-box">${{aiText}}</div>

    <div class="stitle">📊 關鍵判斷因素 (SHAP 分析)</div>
    ${{factors}}

    <div class="stitle">📈 綜合評分</div>
    <div class="score-row"><span class="score-lbl">HSI 棲地指數</span><div class="score-bg"><div class="score-fill" style="width:${{d.pct}}%;background:${{d.marker_color}}"></div></div><span class="score-v">${{d.hsi_str}}</span></div>
    <div class="score-row"><span class="score-lbl">安全評分</span><div class="score-bg"><div class="score-fill" style="width:${{d.safety_score*100}}%;background:${{sc}}"></div></div><span class="score-v">${{d.safety_score}}</span></div>
    <div class="score-row"><span class="score-lbl">CPUE 漁獲指數</span><div class="score-bg"><div class="score-fill" style="width:${{Math.min(d.cpue_index,100)}}%;background:#00aa88"></div></div><span class="score-v">${{d.cpue_index}}</span></div>
    <div class="score-row"><span class="score-lbl">ML 預測 CPUE</span><div class="score-bg"><div class="score-fill" style="width:${{Math.min(d.ml_cpue_kg_day/30*100,100)}}%;background:#7040a0"></div></div><span class="score-v">${{d.ml_cpue_kg_day}}kg</span></div>

    <div class="stitle">🐟 此區域魚種機率</div>
    ${{sp}}

    <div class="stitle">🧠 全特徵 SHAP 貢獻度</div>
    ${{shap}}

    <div class="stitle">🎣 船長作業建議</div>
    <div style="font-size:13px;color:#5a7a9a;margin-bottom:4px">⏰ 最佳時段: <span style="color:#b89520;font-size:14px;font-weight:600">${{d.best_fishing_time}}</span></div>
    <div style="font-size:13px;color:#5a7a9a;margin-bottom:4px">🎯 漁法: <span style="color:#a0b0c0;font-weight:600">${{d.fishing_method}}</span></div>
    <div class="hook-d"><div class="sh">淺</div><div class="opt">★ ${{d.hook_depth_label}}</div><div class="dp">深</div></div>
    <div style="font-size:11px;color:#3a5a7a;margin-top:4px">${{d.lunar_fishing_impact}}</div>
    <div style="font-size:11px;color:#3a5a7a;margin-top:2px">🛰 GFW驗證: ${{d.gfw_validation}} · ${{d.primary_species_prob}}</div>
  `;

  // TAB: Navigation
  const tyDist=Math.round(Math.sqrt((d.lat-13)**2+(d.lon-139)**2)*111);
  document.getElementById('tab-nav').innerHTML=`
    <div class="stitle">🧭 航線規劃</div>
    <div class="route-grid">
      <div><div class="rv">${{d.distance_km}}km</div><div class="rl">總距離</div></div>
      <div><div class="rv">${{d.transit_days}}天</div><div class="rl">預計航程</div></div>
      <div><div class="rv">$${{Math.round(d.route_cost_usd).toLocaleString()}}</div><div class="rl">預估成本</div></div>
      <div><div class="rv">${{d.fuel_ton}}t</div><div class="rl">燃料需求</div></div>
    </div>
    <div style="font-size:12px;color:#5a7a9a;margin:6px 0">📍 方位 <span style="color:#b89520;font-weight:600;font-size:14px">${{d.bearing}}</span> · 距母港 <span style="color:#00aa88;font-weight:600">${{d.dist_nm}} nm</span></div>
    <div class="alert blue">🧭 航線已繪製在地圖上 (點擊 🧭 航線 按鈕切換顯示)</div>

    <div class="stitle">🌙 月相影響</div>
    <div class="dgrid">
      <div class="dcell"><div class="dl">月相</div><div class="dv" style="font-size:13px">${{d.lunar_phase}}</div></div>
      <div class="dcell"><div class="dl">月光因子</div><div class="dv">${{d.lunar_factor}}</div></div>
    </div>
    <div class="alert blue">${{d.lunar_fishing_impact||'月相適中，無顯著影響'}}</div>

    <div class="stitle">📊 漁獲預估 (Bootstrap)</div>
    <div class="cpue-bar-bg"><div class="cpue-bar-range" style="left:${{d.cpue_ci_low}}%;width:${{Math.max(d.cpue_ci_high-d.cpue_ci_low,5)}}%"></div><div class="cpue-bar-mid" style="left:${{d.cpue_index}}%"></div></div>
    <div style="display:flex;justify-content:space-between;font-size:9px;color:#2a4a6a;font-family:'Share Tech Mono'"><span>P25:${{d.cpue_ci_low}}</span><span>中位:${{d.cpue_index}}</span><span>P75:${{d.cpue_ci_high}}</span></div>

    <div class="stitle">🗺 EEZ 資訊</div>
    <div style="font-size:12px;color:#a0b0c0">${{d.eez}} ${{d.eez_authorized?'✅ 已授權':'❌ 未授權'}}</div>
  `;

  // TAB: Safety
  document.getElementById('tab-saf').innerHTML=`
    <div class="stitle">⚠ 安全總覽</div>
    <div style="display:flex;align-items:center;gap:8px;margin:8px 0">
      <span style="font-size:32px">${{d.safety_emoji}}</span>
      <div><div style="font-size:16px;font-weight:700;color:${{sc}}">${{d.safety_level}}</div><div style="font-size:12px;color:#5a7a9a">${{d.safety_reason}}</div></div>
    </div>

    <div class="stitle">🌊 海流剪切風險</div>
    <div class="alert ${{d.shear_risk_level>=2?'red':d.shear_risk_level>=1?'yellow':'blue'}}">${{d.shear_risk_level>=2?'🔴 高風險':d.shear_risk_level>=1?'🟡 中風險':'🟢 安全'}} — 剪切 ${{d.current_shear_ms}} m/s${{d.shear_warning?' · '+d.shear_warning:''}}</div>

    <div class="stitle">🌬 風速</div>
    <div class="dgrid">
      <div class="dcell"><div class="dl">風速</div><div class="dv">${{d.wind_speed_ms}}<span class="du">m/s</span></div></div>
      <div class="dcell"><div class="dl">狀態</div><div class="dv" style="font-size:14px;color:${{d.wind_warning?'#c83040':'#28a050'}}">${{d.wind_warning?'⚠偏強':'✅正常'}}</div></div>
    </div>

    <div class="stitle">🌀 颱風距離</div>
    <div class="alert ${{tyDist<300?'red':tyDist<600?'yellow':'blue'}}">最近系統: <b>TD-03 熱帶低壓</b><br>距此熱點: <b>${{tyDist}} km</b><br>預判: WNW 方向 · 風速 15m/s<br>${{tyDist<300?'🔴 距離太近，建議規避':tyDist<600?'🟡 持續監視，注意動態':'🟢 距離遠，暫無影響'}}</div>

    <div class="stitle">🛡 規避建議</div>
    ${{d.safety_level==='AVOID'?'<div class="alert red">❌ 此區域不建議作業，請選擇其他熱點</div>':d.safety_level==='CAUTION'?'<div class="alert yellow">⚠ 密切注意海況<br>如有惡化應立即轉移</div>':'<div class="alert blue">✅ 海況良好，適合作業<br>建議正常出航</div>'}}
  `;

  // TAB: Ocean
  const [sC,sL]=sm('sst'),[cC,cL]=sm('chl'),[mC,mL]=sm('mld');
  document.getElementById('tab-ocn').innerHTML=`
    <div class="stitle">🌡 水文數據</div>
    <div class="dgrid">
      <div class="dcell"><div class="dl">SST 海溫</div><div>${{fc(d.sst,'°C',sC,sL)}}</div></div>
      <div class="dcell"><div class="dl">CHL 葉綠素</div><div>${{fc(d.chl,'mg/m³',cC,cL)}}</div></div>
      <div class="dcell"><div class="dl">MLD 混合層</div><div>${{fc(d.mld_m,'m',mC,mL)}}</div></div>
      <div class="dcell"><div class="dl">Z20 溫躍層</div><div>${{fc(d.z20_m,'m','live','HYCOM')}}</div></div>
      <div class="dcell"><div class="dl">DO 溶氧</div><div>${{fc(d.do_surface,'μmol','hist','WOA')}}</div></div>
      <div class="dcell"><div class="dl">水深</div><div>${{fc(d.depth_m,'m','live','ETOPO')}}</div></div>
      <div class="dcell"><div class="dl">SSH 異常</div><div>${{fc(d.ssh_anomaly,'m','live','NRT')}}</div></div>
      <div class="dcell"><div class="dl">渦旋</div><div><span class="dv" style="font-size:13px">${{d.eddy_type||'—'}}</span></div></div>
    </div>
    <div class="stitle">🌊 海流/鋒面</div>
    <div class="dgrid">
      <div class="dcell"><div class="dl">流速</div><div>${{fc(d.current_speed,'m/s','hist','再分析')}}</div></div>
      <div class="dcell"><div class="dl">CHL鋒面</div><div>${{fc(d.chl_front_strength,'','live','即時')}}</div></div>
      <div class="dcell"><div class="dl">匯流度</div><div>${{fc(d.convergence,'','live','FTLE')}}</div></div>
      <div class="dcell"><div class="dl">底水溫</div><div>${{fc(d.bottom_temp_c,'°C','live','HYCOM')}}</div></div>
    </div>
    <div class="stitle">🦐 生態指標</div>
    <div class="dgrid">
      <div class="dcell"><div class="dl">GreenFish HSI</div><div class="dv">${{d.greenfish_hsi}}</div></div>
      <div class="dcell"><div class="dl">攝食指數</div><div class="dv">${{d.feeding_index}}</div></div>
      <div class="dcell"><div class="dl">DVM 深度</div><div class="dv">${{d.dvm_depth_m}}<span class="du">m</span></div></div>
      <div class="dcell"><div class="dl">浮游密度</div><div class="dv">${{d.zoo_proxy}}</div></div>
    </div>
  `;
}}

if(D.length>0)show(D[0].id);
'''
