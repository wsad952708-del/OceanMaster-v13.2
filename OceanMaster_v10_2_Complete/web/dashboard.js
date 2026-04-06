// ═══════════════════════════════════════════════════════════
// OceanMaster v12 — Commercial Maritime Dashboard  [v12-fix]
// 精準座標 · 漁區視覺化 · 清晰分析 · 船長友善
// ═══════════════════════════════════════════════════════════

let allHotspots = [], displayHotspots = [];
let activeSp = 'all', activeTab = 'hotspots';
let seaConditions = {}, typhoonData = [];
let layerState = { hotspots: true, typhoon: false, sst: false };
const MIN_HSI = 0.15;  // v12: adjusted for percentile-rescaled scores (0.15-0.90)

const API_KEY = document.querySelector('meta[name="api-key"]')?.content || 'dev-key-change-me';
const authH = () => ({ 'X-API-Key': API_KEY });

// ── Map ──
const map = L.map('map', { zoomControl: false, attributionControl: false }).setView([25, 132], 5);
L.control.zoom({ position: 'bottomright' }).addTo(map);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19, subdomains: 'abcd'
}).addTo(map);

const zoneLayer = L.layerGroup().addTo(map);       // fishing zone circles
const hotspotLayer = L.layerGroup().addTo(map);     // markers
const labelLayer = L.layerGroup().addTo(map);       // coordinate labels
const typhoonLayer = L.layerGroup();
const sstLayer = L.layerGroup();                    // SST color overlay
const eezLayer = L.layerGroup().addTo(map);          // EEZ boundaries

// ── Clock ──
function tick() {
    const n = new Date(), p = x => String(x).padStart(2, '0');
    document.getElementById('clock').textContent =
        `${p(n.getHours())}:${p(n.getMinutes())}:${p(n.getSeconds())} UTC+8`;
}
tick(); setInterval(tick, 1000);

// ── Species ──
const SP = {
    bigeye: { n: '大目鮪', ic: '🐠', temp: [15, 22], desc: '中深層鮪魚，偏好冷水上升流帶，黑潮交匯區' },
    yellowfin: { n: '黃鰭鮪', ic: '🐟', temp: [22, 30], desc: '表層暖水魚，常出沒於海溫鋒面帶' },
    skipjack: { n: '正鰹', ic: '🐡', temp: [24, 30], desc: '表層群游魚，偏好高生產力水域' },
    albacore: { n: '長鰭鮪', ic: '🎣', temp: [14, 20], desc: '冷水域鮪魚，棲息溫度鋒面邊緣' },
    swordfish: { n: '旗魚', ic: '⚔️', temp: [18, 25], desc: '深水洄游種，偏好陡峭海底地形' },
    squid: { n: '魷魚', ic: '🦑', temp: [15, 24], desc: '偏好上升流帶，夜間趨光聚集' },
};
const FL = {
    sst: '海表溫度', chl: '葉綠素', ssh: '海面高度', 'do': '溶解氧',
    current_speed: '海流速度', front_strength: '鋒面強度', eddy_strength: '渦旋強度',
    phi: '代謝指數Φ', bathy_depth: '海底深度', bathy_slope: '海底坡度',
    dist_seamount: '距海山', dist_shelf_break: '距陸棚邊', eke: '渦動動能', npp: '初級生產力',
};

// ═══════════════════════════════════════════════════
//  Smart Analysis — 根據真實環境數據
// ═══════════════════════════════════════════════════
function analyze(h) {
    const sp = h.species || 'unknown';
    const s = SP[sp] || { n: sp, ic: '🐟', temp: [16, 28], desc: '' };
    const sst = h.sst || 0, npp = h.npp || 0, doS = h.do_surface || 0;
    const depth = h.depth_m || 0, phi = h.phi || 0, eke = h.eke || 0;
    const feed = h.feeding_index || 0, z20 = h.z20_m || 0, mld = h.mld_m || 0;
    const gf = h.greenfish_hsi || 0, score = h.score || 0;
    const reasons = [], tags = [];

    if (sst > 0) {
        const [lo, hi] = s.temp;
        if (sst >= lo && sst <= hi) {
            reasons.push(`🌡️ 海溫 ${sst.toFixed(1)}°C，落在${s.n}最適範圍 ${lo}-${hi}°C，代謝率與覓食活力最佳`);
            tags.push(`SST ${sst.toFixed(1)}°C ✓`);
        } else {
            reasons.push(`🌡️ 海溫 ${sst.toFixed(1)}°C (${s.n}適溫 ${lo}-${hi}°C)`);
        }
    }
    if (npp > 1500) {
        reasons.push(`🌱 初級生產力 NPP = ${npp.toFixed(0)} mgC/m²/天，浮游植物旺盛，食物鏈底層豐富`);
        tags.push(`NPP ${npp.toFixed(0)}`);
    }
    if (doS > 4.5) {
        reasons.push(`💧 溶解氧 ${doS.toFixed(1)} ml/L + 代謝指數 Φ = ${phi.toFixed(1)}，氧氣充足魚類可正常棲息`);
        tags.push(`DO ${doS.toFixed(1)}`);
    }
    if (depth > 0) {
        const dl = depth > 3000 ? '深海大洋' : depth > 1000 ? '大陸斜坡' : depth > 200 ? '陸棚邊緣' : '淺海';
        reasons.push(`⛰️ 水深 ${depth.toFixed(0)}m (${dl})，${depth > 1000 ? '深水上升流帶來營養鹽' : '地形上升流效應'}`);
        tags.push(`${depth.toFixed(0)}m`);
    }
    if (feed > 0.4) {
        reasons.push(`🍽️ 覓食指數 ${(feed * 100).toFixed(0)}%，SEAPODYM 判定此區餌料充足`);
        tags.push(`覓食${(feed * 100).toFixed(0)}%`);
    }
    if (z20 > 150) reasons.push(`📏 溫躍層 ${z20.toFixed(0)}m，暖水層厚，鮪魚活動空間充足`);
    if (mld > 50) reasons.push(`🌊 混合層 ${mld.toFixed(0)}m，營養鹽充分混合`);
    if (gf > 25) reasons.push(`🐟 GreenFish HSI ${gf.toFixed(0)}%，獨立模型驗證此棲息地適合`);
    if (eke > 0.001) { reasons.push(`🌀 渦動動能偏高，渦旋邊緣營養上湧，CPUE 可提高 3-5 倍`); tags.push('渦旋'); }

    const cl = score >= 0.85 ? '🟢 極高' : score >= 0.7 ? '🟡 高' : score >= 0.5 ? '🟠 中等' : '🔵 偏低';
    return { reasons, tags, cl, s };
}

function parseShap(h) {
    const sv = h.shap_values;
    if (!sv || typeof sv !== 'object') return [];
    const items = Object.entries(sv)
        .map(([k, v]) => ({ key: k, val: +v, abs: Math.abs(+v), label: FL[k] || k }))
        .sort((a, b) => b.abs - a.abs).slice(0, 6);
    const tot = items.reduce((s, x) => s + x.abs, 0) || 1;
    return items.map(x => ({ ...x, pct: Math.round(x.abs / tot * 100), pos: x.val > 0 }));
}

// ═══════════════════════════════════════════════════
//  Data Loading
// ═══════════════════════════════════════════════════
async function loadAll() {
    try { await Promise.all([loadHotspots(), loadSea()]); } catch (e) { console.error(e); }
    const ld = document.getElementById('loadingOverlay');
    ld?.classList.add('hide'); setTimeout(() => ld?.remove(), 500);
}

async function loadHotspots() {
    try {
        const r = await fetch('/api/hotspots', { headers: authH() });
        if (!r.ok) throw new Error(r.status);
        const d = await r.json();
        allHotspots = d.hotspots || [];
        typhoonData = d.typhoons || [];

        // [v10.5] Data freshness display
        const freshEl = document.getElementById('dataFreshness');
        if (freshEl && d.data_freshness != null) {
            const hrs = d.data_freshness;
            freshEl.textContent = hrs < 1 ? `${Math.round(hrs * 60)}分前更新` :
                hrs < 24 ? `${hrs.toFixed(1)}小時前更新` : `${Math.round(hrs / 24)}天前更新`;
            freshEl.style.color = hrs < 6 ? 'var(--ok)' : hrs < 24 ? 'var(--warn)' : 'var(--danger)';
        }

        displayHotspots = allHotspots
            .filter(h => (h.score || 0) >= MIN_HSI)
            .sort((a, b) => (b.score || 0) - (a.score || 0));
        displayHotspots.forEach((h, i) => h._rank = i + 1);
        updateStats(); renderMap(); renderSidebar(); renderTyphoon();
        setStatus('ok', '數據已更新');
    } catch (e) { console.warn(e); setStatus('warn', '等待數據...'); }
}

async function loadSea() {
    try {
        const r = await fetch('/api/sea-conditions', { headers: authH() });
        if (!r.ok) return;
        seaConditions = await r.json();  // v10.5: server returns flat object, no .conditions wrapper
        renderSeaGrid();
    } catch (e) { }
}

function updateStats() {
    const hs = displayHotspots;
    document.getElementById('qsN').textContent = hs.length;
    document.getElementById('sbCnt').textContent = hs.length;
    if (hs.length) {
        document.getElementById('qsHSI').textContent = Math.round((hs[0].score || 0) * 100) + '%';
        const avg = hs.reduce((s, h) => s + (h.sst || 0), 0) / hs.length;
        if (avg > 0) document.getElementById('qsSST').textContent = avg.toFixed(1) + '°C';
    }
    const strip = document.getElementById('alertStrip');
    const filtered = allHotspots.length - displayHotspots.length;
    if (typhoonData.length) {
        strip.className = 'typhoon'; strip.style.display = 'block';
        strip.innerHTML = `🌀 <b>颱風警報</b> — 點擊左側 🌀 查看詳情與影響範圍`;
        document.getElementById('tyBadge').style.display = 'flex';
        document.getElementById('tyBadge').textContent = typhoonData.length;
    } else if (filtered > 0 && hs.length > 0) {
        strip.className = 'info'; strip.style.display = 'block';
        strip.innerHTML = `📡 已過濾 ${filtered} 個低品質結果 · 僅顯示 <b>${hs.length}</b> 個高可信度漁場 (HSI ≥ ${Math.round(MIN_HSI * 100)}%)`;
    } else strip.style.display = 'none';
}

function setStatus(t, s) {
    document.getElementById('stDot').className = 'st-dot ' + t;
    document.getElementById('stTxt').textContent = s;
}

// ═══════════════════════════════════════════════════
//  Map Rendering — 漁區圓 + 精準標籤
// ═══════════════════════════════════════════════════
function renderMap() {
    hotspotLayer.clearLayers();
    labelLayer.clearLayers();
    zoneLayer.clearLayers();
    const spots = getFiltered();
    const labelPositions = []; // for collision avoidance

    spots.forEach((h, i) => {
        const lat = h.lat || 0, lon = h.lon || 0, score = h.score || 0;
        const sp = h.species || 'unknown';
        const s = SP[sp] || { n: sp, ic: '🐟' };
        const rank = h._rank || (i + 1);
        const pct = Math.round(score * 100);
        const an = analyze(h);

        const color = score >= 0.75 ? '#10b981' : score >= 0.55 ? '#22d3ee' : score >= 0.35 ? '#f59e0b' : '#6b7280';
        const sz = rank === 1 ? 40 : rank <= 3 ? 32 : rank <= 5 ? 26 : 20;
        const glw = rank === 1 ? `0 0 28px ${color}70, 0 0 50px ${color}25` : `0 0 10px ${color}40`;

        // ★ Fishing zone radius (visual) — 大圓顯示漁場範圍
        const zoneR = rank === 1 ? 40000 : rank <= 3 ? 30000 : 25000; // meters
        zoneLayer.addLayer(L.circle([lat, lon], {
            radius: zoneR, color, weight: 1.5, opacity: 0.2 + (score * 0.3),
            fillColor: color, fillOpacity: 0.03 + (rank === 1 ? 0.04 : 0), dashArray: rank === 1 ? null : '6,6'
        }));

        // Marker
        const icon = L.divIcon({
            className: 'custom-marker',
            html: `<div style="
                width:${sz}px;height:${sz}px;border-radius:50%;
                background:radial-gradient(circle at 35% 35%,${color}ee,${color}88);
                border:${rank === 1 ? 3 : 2}px solid ${color};
                box-shadow:${glw};
                display:flex;align-items:center;justify-content:center;
                font-size:${rank === 1 ? 16 : sz > 26 ? 13 : 11}px;color:#fff;font-weight:900;
                cursor:pointer;transition:all .2s;
                ${rank === 1 ? 'animation:pulse 2s infinite;' : ''}
            " onmouseover="this.style.transform='scale(1.3)'" onmouseout="this.style.transform='scale(1)'"
            >${rank}</div>
            ${rank === 1 ? '<style>@keyframes pulse{0%,100%{box-shadow:' + glw + '}50%{box-shadow:0 0 40px ' + color + '90,0 0 70px ' + color + '40}}</style>' : ''}`,
            iconSize: [sz, sz], iconAnchor: [sz / 2, sz / 2],
        });

        const marker = L.marker([lat, lon], { icon });

        // Popup
        const tagHtml = an.tags.map(t =>
            `<span style="display:inline-block;padding:3px 8px;border-radius:5px;font-size:10px;font-weight:600;background:${color}15;color:${color};border:1px solid ${color}25;margin:2px">${t}</span>`
        ).join(' ');

        marker.bindPopup(`
        <div style="min-width:300px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <span style="font-size:16px;font-weight:800">${s.ic} #${rank} ${s.n}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:900;color:${color}">${pct}%</span>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:#22d3ee;margin:6px 0;padding:8px 12px;background:rgba(0,0,0,.3);border-radius:8px;border-left:3px solid ${color}">
                📍 ${lat.toFixed(2)}°N &nbsp; ${lon.toFixed(2)}°E
            </div>
            <div style="font-size:11px;color:#7a8ca8;margin:6px 0">
                ${an.cl} · 距離 ${(h.distance_nm || 0).toFixed(0)} 海浬 · 航程約 ${(h.travel_hours || 0).toFixed(0)} 小時
            </div>
            <div style="margin:8px 0">${tagHtml}</div>
            <div style="border-top:1px solid rgba(255,255,255,.06);padding-top:8px;margin-top:8px">
                <div style="font-size:12px;font-weight:700;color:#00d4ff;margin-bottom:6px">📊 為什麼這裡有 ${s.n} ?</div>
                ${an.reasons.slice(0, 3).map(r => `<div style="margin:4px 0;font-size:12px;line-height:1.6;color:#b0c0d8">• ${r}</div>`).join('')}
            </div>
            <div style="text-align:center;margin-top:10px">
                <span onclick="openShap(${i})" style="color:#00d4ff;cursor:pointer;font-size:12px;font-weight:700">🔬 完整 AI 分析報告 →</span>
            </div>
        </div>`, { maxWidth: 380 });
        hotspotLayer.addLayer(marker);

        // ★ Map coordinate labels — collision avoidance
        let labelLat = lat, labelLon = lon;
        let hasCollision = false;
        for (const prev of labelPositions) {
            if (Math.abs(prev.lat - lat) < 0.8 && Math.abs(prev.lon - lon) < 1.5) {
                hasCollision = true; break;
            }
        }
        labelPositions.push({ lat, lon });

        // Only show label if no collision or it's #1
        if (!hasCollision || rank === 1) {
            const offsetY = rank === 1 ? -(sz / 2 + 4) : -(sz / 2 + 2);
            const lblIcon = L.divIcon({
                className: '',
                html: `<div class="map-lbl ${rank === 1 ? 'best' : ''}">
                    ${rank === 1 ? '🏆 ' : ''}#${rank} ${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E
                    <small>${s.ic} ${s.n} · ${pct}%</small>
                </div>`,
                iconSize: [0, 0],
                iconAnchor: [-8, -offsetY],
            });
            labelLayer.addLayer(L.marker([lat, lon], { icon: lblIcon, interactive: false }));
        }
    });

    // SST color reference zones (when SST layer is on)
    renderSSTOverlay();

    if (spots.length > 0) {
        const bounds = L.latLngBounds(spots.map(h => [h.lat, h.lon]));
        map.fitBounds(bounds.pad(0.5), { maxZoom: 7 });
    }
}

// ═══════════════════════════════════════════════════
//  SST Color Overlay — 根據熱點 SST 畫色溫圈
// ═══════════════════════════════════════════════════
function renderSSTOverlay() {
    sstLayer.clearLayers();
    if (!layerState.sst) return;
    const spots = getFiltered();
    spots.forEach(h => {
        const sst = h.sst || 0;
        if (sst <= 0) return;
        // Color by temperature
        const hue = sst < 16 ? 240 : sst < 20 ? 200 : sst < 24 ? 160 : sst < 28 ? 40 : 0;
        const col = `hsl(${hue}, 70%, 50%)`;
        sstLayer.addLayer(L.circle([h.lat, h.lon], {
            radius: 25000, color: col, weight: 1,  // v10.5: reduced from 60km to 25km
            fillColor: col, fillOpacity: 0.08
        }));
    });
}

// ═══════════════════════════════════════════════════
//  Sidebar — Cards
// ═══════════════════════════════════════════════════
function renderSidebar() {
    if (activeTab === 'hotspots') renderCards();
    else if (activeTab === 'ranking') renderRanking();
    else renderAnalysis();
}

function renderCards() {
    const el = document.getElementById('sbContent');
    const spots = getFiltered();
    if (!spots.length) {
        el.innerHTML = `<div style="text-align:center;padding:50px 20px;color:var(--txt3)">
            <div style="font-size:36px;margin-bottom:10px">📡</div>
            <div style="font-size:14px;font-weight:600">暫無高品質漁場資料</div>
            <div style="font-size:11px;margin-top:4px">已過濾 HSI < ${Math.round(MIN_HSI * 100)}% 結果</div>
        </div>`;
        return;
    }

    let html = '';

    // ═══ #1 BEST Card ═══
    const b = spots[0];
    const bs = b.score || 0;
    const bsp = b.species || 'unknown';
    const bi = SP[bsp] || { n: bsp, ic: '🐟', desc: '' };
    const ba = analyze(b);
    const bpct = Math.round(bs * 100);
    const bcolor = bs >= 0.75 ? 'var(--ok)' : bs >= 0.55 ? 'var(--cyan)' : 'var(--warn)';

    html += `
    <div class="best-card" onclick="flyTo(${b.lat},${b.lon},0)">
        <div class="best-badge">🏆 最佳推薦</div>

        <div class="best-header">
            <span class="best-icon">${bi.ic}</span>
            <div>
                <div class="best-sp">${bi.n}</div>
                <div style="font-size:11px;color:var(--txt3)">${bi.desc}</div>
            </div>
            <div style="margin-left:auto;text-align:right">
                <div class="best-score" style="color:${bcolor}">${bpct}%</div>
                <div class="best-score-lbl">HSI 信心度</div>
            </div>
        </div>

        <div class="best-coord">
            <small>📍 精準座標 (建議航向)</small>
            ${(b.lat || 0).toFixed(2)}°N &nbsp; ${(b.lon || 0).toFixed(2)}°E
        </div>

        <div class="best-metrics">
            <div class="bm"><div class="lab">🌡️ 海表溫度</div><div class="val">${(b.sst || 0).toFixed(1)}°C</div></div>
            <div class="bm"><div class="lab">💧 溶解氧</div><div class="val">${(b.do_surface || 0).toFixed(1)} ml/L</div></div>
            <div class="bm"><div class="lab">🌱 初級生產力</div><div class="val">${(b.npp || 0).toFixed(0)} mgC</div></div>
            <div class="bm"><div class="lab">🧬 代謝指數 Φ</div><div class="val">${(b.phi || 0).toFixed(1)}</div></div>
            <div class="bm"><div class="lab">⛰️ 水深</div><div class="val">${(b.depth_m || 0).toFixed(0)} m</div></div>
            <div class="bm"><div class="lab">🍽️ 覓食指數</div><div class="val">${((b.feeding_index || 0) * 100).toFixed(0)}%</div></div>
            <div class="bm"><div class="lab">📏 距離</div><div class="val">${(b.distance_nm || 0).toFixed(0)} 海浬</div></div>
            <div class="bm"><div class="lab">⏱️ 航程</div><div class="val">${(b.travel_hours || 0).toFixed(0)} 小時</div></div>
        </div>

        <div class="best-reasons">
            <div class="title">📊 為什麼這裡有 ${bi.n} ?</div>
            ${ba.reasons.slice(0, 4).map(r => `<div class="r">• ${r}</div>`).join('')}
        </div>

        <div class="best-link" onclick="event.stopPropagation();openShap(0)">🔬 查看完整 AI 分析報告 →</div>
    </div>`;

    // ═══ Remaining Cards ═══
    spots.slice(1).forEach((h, i) => {
        const idx = i + 1;
        const score = h.score || 0;
        const sp = h.species || 'unknown';
        const s = SP[sp] || { n: sp, ic: '🐟' };
        const rank = h._rank || (idx + 1);
        const pct = Math.round(score * 100);
        const an = analyze(h);
        const col = score >= 0.75 ? 'var(--ok)' : score >= 0.55 ? 'var(--cyan)' : 'var(--warn)';
        const tier = score >= 0.55 ? 'tier-s' : 'tier-a';
        const rc = rank === 2 ? 'r2' : rank === 3 ? 'r3' : 'rn';

        html += `
        <div class="h-card ${tier}" onclick="flyTo(${h.lat},${h.lon},${idx})">
            <div class="hc-top">
                <div style="display:flex;align-items:center;gap:10px">
                    <div class="hc-rank ${rc}">${rank}</div>
                    <div>
                        <div class="hc-sp">${s.ic} ${s.n}</div>
                        <div class="hc-coord">${(h.lat || 0).toFixed(2)}°N, ${(h.lon || 0).toFixed(2)}°E</div>
                    </div>
                </div>
                <div class="hc-score-area">
                    <div class="hc-pct ${score >= 0.85 ? 'hi' : score >= 0.7 ? 'mid' : 'lo'}">${pct}%</div>
                    <div class="hc-bar"><i style="width:${pct}%;background:${col}"></i></div>
                </div>
            </div>
            <div class="hc-grid">
                <span>🌡️ SST <b>${(h.sst || 0).toFixed(1)}°C</b></span>
                <span>💧 DO <b>${(h.do_surface || 0).toFixed(1)} ml/L</b></span>
                <span>📏 距離 <b>${(h.distance_nm || 0).toFixed(0)} 海浬</b></span>
                <span>⏱️ 航程 <b>${(h.travel_hours || 0).toFixed(0)} 小時</b></span>
            </div>
            <div class="hc-why">📊 ${an.reasons[0] || ''}</div>
            <div class="hc-link" onclick="event.stopPropagation();openShap(${idx})">🔬 完整分析 →</div>
        </div>`;
    });

    el.innerHTML = html;
}

function renderRanking() {
    const el = document.getElementById('sbContent');
    const spots = getFiltered();
    el.innerHTML = `<div style="padding:8px 4px">
        <div style="font-size:14px;font-weight:800;margin-bottom:12px">🏆 漁場排行 (HSI ≥ ${Math.round(MIN_HSI * 100)}%)</div>
        ${spots.map((h, i) => {
        const score = h.score || 0, pct = Math.round(score * 100);
        const sp = h.species || 'unknown';
        const s = SP[sp] || { n: sp, ic: '🐟' };
        const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i + 1}`;
        const col = score >= 0.75 ? 'var(--ok)' : score >= 0.55 ? 'var(--cyan)' : 'var(--warn)';
        return `
            <div style="display:flex;align-items:center;gap:12px;padding:12px 4px;border-bottom:1px solid var(--bdr);cursor:pointer" onclick="flyTo(${h.lat},${h.lon},${i})">
                <span style="font-size:20px;width:32px;text-align:center">${medal}</span>
                <div style="flex:1">
                    <div style="font-size:13px;font-weight:700">${s.ic} ${s.n}</div>
                    <div style="font-family:var(--mono);font-size:13px;color:var(--cyan);margin-top:2px">${(h.lat || 0).toFixed(2)}°N, ${(h.lon || 0).toFixed(2)}°E</div>
                    <div style="height:7px;background:var(--bg-1);border-radius:4px;margin-top:5px;overflow:hidden">
                        <div style="height:100%;width:${pct}%;background:${col};border-radius:4px;transition:width .6s"></div>
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="font-family:var(--mono);font-size:22px;font-weight:900;color:${col}">${pct}%</div>
                    <div style="font-size:9px;color:var(--txt3)">HSI</div>
                </div>
            </div>`;
    }).join('')}
    </div>`;
}

function renderAnalysis() {
    const el = document.getElementById('sbContent');
    const spots = getFiltered();
    const spDist = {};
    spots.forEach(h => {
        const sp = h.species || 'unknown';
        if (!spDist[sp]) spDist[sp] = { n: 0, ts: 0, bs: 0, blat: 0, blon: 0 };
        spDist[sp].n++; spDist[sp].ts += (h.score || 0);
        if ((h.score || 0) > spDist[sp].bs) { spDist[sp].bs = h.score; spDist[sp].blat = h.lat; spDist[sp].blon = h.lon; }
    });
    const avgSST = spots.reduce((s, h) => s + (h.sst || 0), 0) / (spots.length || 1);
    const avgDO = spots.reduce((s, h) => s + (h.do_surface || 0), 0) / (spots.length || 1);
    const avgNPP = spots.reduce((s, h) => s + (h.npp || 0), 0) / (spots.length || 1);

    el.innerHTML = `<div style="padding:8px 4px">
        <div style="font-size:14px;font-weight:800;margin-bottom:12px">📊 綜合分析報告</div>

        <div style="background:var(--bg-card);border:1px solid var(--bdr);border-radius:var(--r);padding:14px;margin-bottom:10px">
            <div style="font-size:12px;font-weight:700;margin-bottom:10px">🐟 魚種分佈與最佳位置</div>
            ${Object.entries(spDist).map(([sp, d]) => {
        const s = SP[sp] || { n: sp, ic: '🐟' };
        return `<div style="padding:6px 0;border-bottom:1px solid rgba(255,255,255,.03)">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:13px;font-weight:700">${s.ic} ${s.n} × ${d.n}</span>
                        <span style="font-family:var(--mono);font-size:15px;font-weight:800;color:var(--accent)">${Math.round(d.ts / d.n * 100)}%</span>
                    </div>
                    <div style="font-family:var(--mono);font-size:11px;color:var(--cyan);margin-top:2px">
                        最佳: ${d.blat.toFixed(2)}°N, ${d.blon.toFixed(2)}°E (${Math.round(d.bs * 100)}%)
                    </div>
                </div>`;
    }).join('')}
        </div>

        <div style="background:var(--bg-card);border:1px solid var(--bdr);border-radius:var(--r);padding:14px;margin-bottom:10px">
            <div style="font-size:12px;font-weight:700;margin-bottom:10px">🌡️ 環境概況</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;text-align:center">
                <div style="padding:8px;background:rgba(0,212,255,.03);border-radius:8px"><div style="font-family:var(--mono);font-size:17px;font-weight:800;color:var(--accent)">${avgSST.toFixed(1)}°C</div><div style="font-size:9px;color:var(--txt3)">平均 SST</div></div>
                <div style="padding:8px;background:rgba(0,212,255,.03);border-radius:8px"><div style="font-family:var(--mono);font-size:17px;font-weight:800;color:var(--accent)">${avgDO.toFixed(1)}</div><div style="font-size:9px;color:var(--txt3)">平均 DO</div></div>
                <div style="padding:8px;background:rgba(0,212,255,.03);border-radius:8px"><div style="font-family:var(--mono);font-size:17px;font-weight:800;color:var(--accent)">${avgNPP.toFixed(0)}</div><div style="font-size:9px;color:var(--txt3)">NPP</div></div>
            </div>
        </div>

        <div style="background:var(--bg-card);border:1px solid var(--bdr);border-radius:var(--r);padding:14px;font-size:12px;color:var(--txt2);line-height:1.8">
            <div style="font-size:12px;font-weight:700;margin-bottom:8px">🛡️ 數據品質</div>
            ✅ 顯示: <b>${displayHotspots.length}</b> 個高品質漁場 (HSI ≥ ${Math.round(MIN_HSI * 100)}%)<br>
            ⛔ 已過濾: ${allHotspots.length - displayHotspots.length} 個低品質結果<br>
            📡 資料源: 30+ (NOAA / CMEMS / HYCOM / Open-Meteo)<br>
            🤖 SHAP 可解釋 AI 已啟用<br>
            🧬 代謝指數 Φ / GreenFish HSI 雙模型驗證
        </div>
    </div>`;
}

// ═══════════════════════════════════════════════════
//  SHAP Modal — Complete Analysis Report
// ═══════════════════════════════════════════════════
function openShap(idx) {
    const h = getFiltered()[idx];
    if (!h) return;
    const sp = h.species || 'unknown';
    const s = SP[sp] || { n: sp, ic: '🐟', desc: '' };
    const score = h.score || 0, pct = Math.round(score * 100);
    const an = analyze(h);
    const shap = parseShap(h);
    const col = score >= 0.85 ? 'var(--ok)' : score >= 0.7 ? 'var(--cyan)' : 'var(--warn)';

    document.getElementById('shapTitle').textContent = `🔬 ${s.ic} ${s.n} — 精準分析報告`;

    const shapHtml = shap.map(x => `
        <div class="shap-bar">
            <div class="shap-lbl">${x.label}</div>
            <div class="shap-track"><div class="shap-fill ${x.pos ? 'pos' : 'neg'}" style="width:${Math.min(85, Math.max(6, x.pct))}%"></div></div>
            <span class="shap-pct">${x.pct}%</span>
        </div>`).join('');

    document.getElementById('shapBody').innerHTML = `
        <!-- Header -->
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px">
            <div style="text-align:center">
                <div style="font-family:var(--mono);font-size:42px;font-weight:900;color:${col};text-shadow:0 0 24px ${col}40">${pct}%</div>
                <div style="font-size:10px;color:var(--txt3)">HSI 信心度</div>
            </div>
            <div style="flex:1">
                <div style="font-size:14px;font-weight:800">${s.ic} ${s.n}</div>
                <div style="font-family:var(--mono);font-size:16px;color:var(--cyan);margin:4px 0;letter-spacing:-.2px">📍 ${(h.lat || 0).toFixed(2)}°N, ${(h.lon || 0).toFixed(2)}°E</div>
                <div style="font-size:11px;color:var(--txt3)">${an.cl} · ${s.desc}</div>
            </div>
        </div>

        <!-- SHAP -->
        <div style="font-size:13px;font-weight:700;margin-bottom:6px">🔍 AI 影響因子分析 (SHAP)</div>
        <div style="font-size:10px;color:var(--txt3);margin-bottom:10px">🟩 綠色 = 正面因子 (推高評分)  &nbsp; 🟥 紅色 = 負面因子</div>
        ${shapHtml}

        <!-- Why Fish Here -->
        <div class="why-box">
            <h4>💡 為什麼這裡有 ${s.n} ?</h4>
            ${an.reasons.length ? an.reasons.map(r => `<div class="r">• ${r}</div>`).join('') : '<div style="color:var(--txt3)">尚無足夠數據生成分析</div>'}
        </div>

        <!-- [v12-enhance] Food Chain Visualization -->
        <div style="margin:12px 0;padding:10px;border-radius:8px;background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.15)">
            <div style="font-size:13px;font-weight:700;margin-bottom:8px">🧪 v12 生態鏈指標</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:11px">
                <div style="display:flex;align-items:center;gap:6px">
                    <span style="font-size:14px">🌙</span>
                    <div>月相 <b style="color:${(h.lunar_illumination||0)>0.5?'var(--warn)':'var(--ok)'}">${h.lunar_phase||'未知'}</b>
                    <div style="font-size:9px;color:var(--txt3)">光照 ${((h.lunar_illumination||0)*100).toFixed(0)}%</div></div>
                </div>
                <div style="display:flex;align-items:center;gap:6px">
                    <span style="font-size:14px">⬇️</span>
                    <div>DVM 深度 <b style="color:var(--cyan)">${(h.dvm_depth_m||0).toFixed(0)}m</b>
                    <div style="font-size:9px;color:var(--txt3)">微小族養層深度</div></div>
                </div>
                <div style="display:flex;align-items:center;gap:6px">
                    <span style="font-size:14px">🧫</span>
                    <div>浮遊動物 <b style="color:${(h.zoo_proxy||0)>0.5?'var(--ok)':'var(--txt3)'}">${((h.zoo_proxy||0)*100).toFixed(0)}%</b>
                    <div style="font-size:9px;color:var(--txt3)">餉料豐度代理</div></div>
                </div>
                <div style="display:flex;align-items:center;gap:6px">
                    <span style="font-size:14px">🫧</span>
                    <div>OMZ 效應 <b style="color:${(h.omz_net_effect||0)>0.3?'var(--warn)':'var(--ok)'}">${((h.omz_net_effect||0)*100).toFixed(0)}%</b>
                    <div style="font-size:9px;color:var(--txt3)">棲息壓縮+邊緣增益</div></div>
                </div>
            </div>
        </div>

        <!-- Env Grid -->
        <div class="env-grid">
            <span>📍 座標 <b>${(h.lat || 0).toFixed(3)}°N, ${(h.lon || 0).toFixed(3)}°E</b></span>
            <span>📏 距離 <b>${(h.distance_nm || 0).toFixed(0)} 海浬</b></span>
            <span>🌡️ SST <b>${(h.sst || 0).toFixed(1)}°C</b></span>
            <span>💧 DO <b>${(h.do_surface || 0).toFixed(1)} ml/L</b></span>
            <span>⛰️ 水深 <b>${(h.depth_m || 0).toFixed(0)} m</b></span>
            <span>🧬 Φ <b>${(h.phi || 0).toFixed(1)}</b></span>
            <span>🌱 NPP <b>${(h.npp || 0).toFixed(0)} mgC/m²/d</b></span>
            <span>🍽️ 覓食 <b>${((h.feeding_index || 0) * 100).toFixed(0)}%</b></span>
            <span>📏 Z20 <b>${(h.z20_m || 0).toFixed(0)} m</b></span>
            <span>🌊 MLD <b>${(h.mld_m || 0).toFixed(0)} m</b></span>
            <span>🌊 EEZ <b>${h.eez_high_seas ? '公海' : (h.eez || '--')}</b></span>
            <span>⏱️ 航程 <b>${(h.travel_hours || 0).toFixed(0)} 小時</b></span>
        </div>`;

    document.getElementById('shapModal').classList.add('show');
}

function closeModal() { document.getElementById('shapModal').classList.remove('show'); }
document.getElementById('shapModal').addEventListener('click', e => { if (e.target.id === 'shapModal') closeModal(); });

// ═══════════════════════════════════════════════════
//  Sea / Typhoon
// ═══════════════════════════════════════════════════
function renderSeaGrid() {
    const c = seaConditions;
    document.getElementById('seaGrid').innerHTML = [
        { l: 'SST Min', v: c.sst_min != null ? c.sst_min.toFixed(1) + '°C' : '--' },
        { l: 'SST Max', v: c.sst_max != null ? c.sst_max.toFixed(1) + '°C' : '--' },
        { l: 'ENSO ONI', v: c.oni != null ? (c.oni >= 0 ? '+' : '') + c.oni.toFixed(2) : '--' },
        { l: 'Wind', v: c.wind_speed != null ? c.wind_speed.toFixed(0) + ' m/s' : '--' },
        { l: 'Current', v: c.current_speed != null ? c.current_speed.toFixed(2) + ' m/s' : '--' },
        { l: 'Moon', v: c.moon_phase || '--' },
    ].map(x => `<div class="sea-it"><div class="v">${x.v}</div><div class="l">${x.l}</div></div>`).join('');
}

function renderTyphoon() {
    const body = document.getElementById('tyBody');
    if (!typhoonData.length) { body.innerHTML = '<div style="text-align:center;padding:20px;color:var(--txt3)">✅ 無颱風警報，海域安全</div>'; return; }
    typhoonLayer.clearLayers();
    typhoonData.forEach(t => {
        const lat = t.lat || t.center_lat || 0, lon = t.lon || t.center_lon || 0;
        const r = (t.danger_radius_km || 300) * 1000;
        typhoonLayer.addLayer(L.circle([lat, lon], { radius: r, color: '#ef4444', weight: 2, opacity: .5, fillColor: '#ef4444', fillOpacity: .06, dashArray: '8,6' }));
        typhoonLayer.addLayer(L.circle([lat, lon], { radius: r * 0.4, color: '#dc2626', weight: 2, opacity: .7, fillColor: '#dc2626', fillOpacity: .12 }));
        const icon = L.divIcon({ className: '', html: '<div style="font-size:32px;animation:spin 2.5s linear infinite">🌀</div>', iconSize: [32, 32], iconAnchor: [16, 16] });
        typhoonLayer.addLayer(L.marker([lat, lon], { icon }));
    });
    body.innerHTML = typhoonData.map(t => `
        <div style="font-size:13px;font-weight:700;color:var(--danger);margin-bottom:8px">🌀 ${t.name || '熱帶氣旋'}</div>
        <div class="ty-row"><span>強度</span><span style="color:var(--danger);font-weight:700">${t.category || '--'}</span></div>
        <div class="ty-row"><span>最大風速</span><span>${t.max_wind_kt || '--'} 節</span></div>
        <div class="ty-row"><span>危險半徑</span><span style="color:var(--danger);font-weight:600">${t.danger_radius_km || '--'} km</span></div>
        <div style="margin-top:8px;padding:8px;background:rgba(239,68,68,.06);border-radius:8px;font-size:10px;color:#fca5a5;line-height:1.6">
            ⚠️ 建議避開中心 ${t.danger_radius_km || 300} km 以上
        </div>`).join('');
}

// ═══════════════════════════════════════════════════
//  Controls
// ═══════════════════════════════════════════════════
function toggleSB() {
    const sb = document.getElementById('sidebar'), btn = document.getElementById('sbToggle');
    sb.classList.toggle('collapsed');
    btn.textContent = sb.classList.contains('collapsed') ? '▶' : '◀';
}
function switchTab(tab) {
    activeTab = tab;
    document.querySelectorAll('.sb-tab').forEach(el => el.classList.toggle('active', el.dataset.tab === tab));
    renderSidebar();
}
function filterSp(sp) {
    activeSp = sp;
    document.querySelectorAll('.sp-btn').forEach(el => el.classList.toggle('active', el.dataset.sp === sp));
    renderMap(); renderSidebar();
}
function getFiltered() {
    if (activeSp === 'all') return displayHotspots;
    return displayHotspots.filter(h => h.species === activeSp);
}
function flyTo(lat, lon) {
    map.flyTo([lat, lon], 8, { duration: 1 });
    hotspotLayer.eachLayer(l => {
        if (l.getLatLng) {
            const ll = l.getLatLng();
            if (Math.abs(ll.lat - lat) < 0.01 && Math.abs(ll.lng - lon) < 0.01) setTimeout(() => l.openPopup(), 300);
        }
    });
}
function toggleLayer(name) {
    layerState[name] = !layerState[name];
    document.getElementById(`btn${name.charAt(0).toUpperCase() + name.slice(1)}`).classList.toggle('active', layerState[name]);
    if (name === 'hotspots') {
        [hotspotLayer, labelLayer, zoneLayer].forEach(l => layerState[name] ? map.addLayer(l) : map.removeLayer(l));
    }
    if (name === 'typhoon') {
        layerState[name] ? map.addLayer(typhoonLayer) : map.removeLayer(typhoonLayer);
        document.getElementById('tyPanel').style.display = layerState[name] ? 'block' : 'none';
    }
}
function toggleSeaPanel() {
    const p = document.getElementById('seaPanel'), btn = document.getElementById('btnSea');
    const show = p.style.display !== 'block';
    p.style.display = show ? 'block' : 'none';
    btn.classList.toggle('active', show);
}
function toggleSSTLayer() {
    layerState.sst = !layerState.sst;
    document.getElementById('btnSST').classList.toggle('active', layerState.sst);
    document.getElementById('sstLegend').style.display = layerState.sst ? 'block' : 'none';
    if (layerState.sst) {
        renderSSTOverlay();
        map.addLayer(sstLayer);
    } else {
        map.removeLayer(sstLayer);
    }
}
function closeTy() { document.getElementById('tyPanel').style.display = 'none'; }

// ═══════════════════════════════════════════════════
loadAll();
setInterval(loadHotspots, 300000);
