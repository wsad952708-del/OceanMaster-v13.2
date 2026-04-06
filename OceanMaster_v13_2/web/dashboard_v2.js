/**
 * OceanMaster v15.3 — Dashboard v2 JavaScript
 * ========================================================
 * 功能:
 *   1. Leaflet 全螢幕地圖 + 熱力圖 (高雄港為中心)
 *   2. 熱點標記 + 左側邊欄列表
 *   3. 食物鏈時間線 (D3 SVG)
 *   4. 右側快速統計面板
 *   5. SHAP 影響因子模態框
 *   6. 時間滑桿 (模擬)
 *   7. 圖層切換 + 黑潮軸線圖層
 */

(() => {
    'use strict';

    // ── 全域狀態 ──
    const API_KEY = document.querySelector('meta[name="api-key"]')?.content || 'dev-key-change-me';
    const API_HEADERS = { 'X-API-Key': API_KEY, 'Content-Type': 'application/json' };
    const SPECIES_ZH = {
        yellowfin: '黃鰭鮪', bigeye: '大目鮪', skipjack: '正鰹', albacore: '長鰭鮪',
        neon_flying_squid: '赤魷', japanese_flying_squid: '日本魷',
        mahi_mahi: '鬼頭刀', blue_marlin: '旗魚',
        mackerel_scad: '竹筴魚', pacific_saury: '秋刀魚'
    };
    // [v13] 近海物種圖示
    const SPECIES_ICON = {
        mahi_mahi: '🐟', blue_marlin: '🐟', mackerel_scad: '🐟',
        pacific_saury: '🐟', yellowfin: '🐟', bigeye: '🐟',
        skipjack: '🐟', albacore: '🐟',
        neon_flying_squid: '🦑', japanese_flying_squid: '🦑'
    };

    let map, heatLayer, hotspotMarkers = [], selectedHotspot = null;
    let hotspots = [], currentSpecies = 'yellowfin';
    let timelineData = null;
    let kuroshioLayer = null;  // [v13] 黑潮軸線圖層

    // ── 初始化 ──
    document.addEventListener('DOMContentLoaded', init);

    async function init() {
        initMap();
        bindEvents();
        await loadData();
        startAutoRefresh();
    }

    // ═══ MAP ═══
    function initMap() {
        map = L.map('map', {
            center: [22.61, 120.28],  // [v13] 高雄港
            zoom: 7,                   // [v13] 近海模式預設
            zoomControl: true,
            attributionControl: false,
        });

        // 暗色底圖
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 18,
            subdomains: 'abcd',
        }).addTo(map);

        // 熱力圖層 (空資料，待填充)
        heatLayer = L.heatLayer([], {
            radius: 30, blur: 20, maxZoom: 10,
            gradient: { 0.2: '#1e3a5f', 0.5: '#fbbf24', 0.8: '#ef4444', 1.0: '#ff0000' }
        }).addTo(map);

        // [v13] 高雄港標記
        const vesselIcon = L.divIcon({
            className: '',
            html: '<div style="font-size:24px;text-shadow:0 0 8px rgba(255,200,0,0.6)">\u26f4</div>',
            iconSize: [28, 28],
            iconAnchor: [14, 14],
        });
        L.marker([22.61, 120.28], { icon: vesselIcon })
            .addTo(map)
            .bindPopup('<strong>\u2693 \u9ad8\u96c4\u6e2f (\u524d\u93ae\u6f01\u6e2f)</strong><br>22.61\u00b0N, 120.28\u00b0E');

        // [v13] 黑潮軸線圖層 (氣候態)
        const kuroshioAxis = [
            [20.0, 121.5], [21.0, 121.3], [22.0, 121.0],
            [23.0, 121.5], [24.0, 122.0], [25.0, 122.5],
            [26.0, 123.0], [27.0, 124.0], [28.0, 125.5],
            [29.0, 127.0], [30.0, 128.5], [31.0, 130.0],
        ];
        kuroshioLayer = L.polyline(kuroshioAxis, {
            color: '#ff6b35', weight: 3, opacity: 0.7,
            dashArray: '8,6',
        });
        kuroshioLayer.bindPopup('\ud83c\udf0a \u9ed1\u6f6e\u4e3b\u8ef8 (\u6c23\u5019\u614b)');
        kuroshioLayer.addTo(map);
    }

    // ═══ 數據載入 ═══
    async function loadData() {
        try {
            const [hotspotsRes, seaCondRes] = await Promise.all([
                apiFetch('/api/hotspots'),
                apiFetch('/api/sea_conditions').catch(() => null),
            ]);

            hotspots = hotspotsRes?.hotspots || [];

            // [v16.0] Safety caps — 防止 100% 和不合理數值
            hotspots.forEach(h => {
                h.score = Math.min(h.score || 0, 0.95);
                h.hsi = Math.min(h.hsi || 0, 0.95);
                h.npp = Math.min(h.npp || 0, 2500);
                h.feeding_index = Math.min(h.feeding_index || 0, 0.95);
                h.zoo_proxy = Math.min(h.zoo_proxy || 0, 0.95);
            });

            renderHotspotList(hotspots);
            renderHotspotMarkers(hotspots);
            renderHeatmap(hotspots);
            renderQuickStats(hotspots, hotspotsRes, seaCondRes);
            renderStarRatings(hotspots);

            // 載入食物鏈時間線
            loadFoodChainTimeline();

            // 載入月相
            loadMoonPhase();

            document.getElementById('hotspot-count').textContent =
                `${hotspots.length} 個漁場熱點`;
            document.getElementById('stat-last-update').textContent =
                hotspotsRes?.last_update ? formatTime(hotspotsRes.last_update) : '--';
            document.getElementById('stat-freshness').textContent =
                hotspotsRes?.data_freshness ? `${hotspotsRes.data_freshness}h` : '--';

        } catch (err) {
            console.error('Data load error:', err);
            document.getElementById('hotspot-count').textContent = '載入失敗';
        }
    }

    async function apiFetch(path) {
        const res = await fetch(path, { headers: API_HEADERS });
        if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
        return res.json();
    }

    function formatTime(iso) {
        if (!iso) return '--';
        const d = new Date(iso);
        return d.toLocaleString('zh-TW', { hour12: false, month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
    }

    // ═══ HOTSPOT LIST ═══
    function renderHotspotList(hs) {
        const container = document.getElementById('hotspot-list');
        container.innerHTML = '';

        hs.slice(0, 20).forEach((h, i) => {
            const score = h.score || h.hsi || 0;
            const pct = Math.round(score * 100);
            const cls = pct >= 70 ? 'hsi-high' : pct >= 40 ? 'hsi-mid' : 'hsi-low';

            const card = document.createElement('div');
            card.className = 'hotspot-card fade-in';
            card.style.animationDelay = `${i * 0.03}s`;
            card.innerHTML = `
            <div class="hotspot-rank">
                <span class="hotspot-number">#${i + 1}</span>
                <span class="hsi-badge ${cls}">${pct}%</span>
            </div>
            <div class="hotspot-info">
                <span class="label">位置</span>
                <span class="value">${(h.lat || 0).toFixed(1)}°N ${(h.lon || 0).toFixed(1)}°E</span>
                <span class="label">SST</span>
                <span class="value">${h.sst ? h.sst.toFixed(1) + '°C' : '--'}</span>
                <span class="label">物種</span>
                <span class="value">${SPECIES_ZH[h.species] || h.species || '--'}</span>
            </div>
        `;
            card.onclick = () => selectHotspot(h, i);
            container.appendChild(card);
        });
    }

    // ═══ MAP MARKERS ═══
    function renderHotspotMarkers(hs) {
        hotspotMarkers.forEach(m => map.removeLayer(m));
        hotspotMarkers = [];

        if (!document.getElementById('layer-hotspots').checked) return;

        hs.slice(0, 20).forEach((h, i) => {
            const isTop3 = i < 3;
            const isSquid = h.species && h.species.includes('squid');
            const iconHTML = isSquid
                ? `<div class="hotspot-marker ${isTop3 ? 'top3' : ''}" style="background:var(--accent-purple, #8b5cf6); border-color:#c4b5fd;">🦑<br>${i + 1}</div>`
                : `<div class="hotspot-marker ${isTop3 ? 'top3' : ''}">${i + 1}</div>`;
            const icon = L.divIcon({
                className: '',
                html: iconHTML,
                iconSize: isTop3 ? [34, 34] : [28, 28],
                iconAnchor: isTop3 ? [17, 17] : [14, 14],
            });

            const marker = L.marker([h.lat || 0, h.lon || 0], { icon })
                .addTo(map)
                .on('click', () => selectHotspot(h, i));

            const pct = Math.round((h.score || h.hsi || 0) * 100);
            const analysisText = h.analysis ? `<br><span style="color:#94a3b8;font-size:14px;line-height:1.5">${h.analysis}</span>` : '';
            marker.bindPopup(`
            <div style="font-family:Inter,'Noto Sans TC',sans-serif;font-size:16px;max-width:360px;line-height:1.6">
                <strong style="font-size:18px">#${i + 1} ${SPECIES_ZH[h.species] || ''}</strong><br>
                HSI: ${pct}% | SST: ${h.sst ? h.sst.toFixed(1) + '°C' : '--'} | Chl: ${h.chl ? h.chl.toFixed(2) + ' mg/m³' : '--'}<br>
                <span style="font-size:17px;font-weight:600">${(h.lat || 0).toFixed(2)}°N, ${(h.lon || 0).toFixed(2)}°E</span>
                ${analysisText}
            </div>
        `);

            hotspotMarkers.push(marker);
        });
    }

    function renderHeatmap(hs) {
        if (!document.getElementById('layer-heatmap').checked) {
            heatLayer.setLatLngs([]);
            return;
        }
        const points = hs.map(h => [h.lat || 0, h.lon || 0, (h.score || h.hsi || 0) * 100]);
        heatLayer.setLatLngs(points);
    }

    // ═══ SELECT HOTSPOT (詳細面板) ═══
    async function selectHotspot(h, idx) {
        selectedHotspot = { data: h, index: idx };

        // 切換到詳細面板
        document.getElementById('hotspot-list').style.display = 'none';
        document.getElementById('detail-panel').style.display = 'block';

        // 標記活動卡片
        document.querySelectorAll('.hotspot-card').forEach((c, i) => {
            c.classList.toggle('active', i === idx);
        });

        // 填充基本資訊
        const pct = Math.round((h.score || h.hsi || 0) * 100);
        document.getElementById('detail-position').textContent =
            `${(h.lat || 0).toFixed(2)}°N, ${(h.lon || 0).toFixed(2)}°E`;
        document.getElementById('detail-hsi').textContent = `${pct}%`;
        document.getElementById('detail-confidence').textContent =
            h.confidence ? `${Math.round(h.confidence * 100)}%` : '--';

        // 海洋條件
        document.getElementById('detail-sst').innerHTML =
            h.sst ? `${h.sst.toFixed(1)}°C <span class="${h.sst > 24 && h.sst < 30 ? 'status-good' : 'status-warning'}">
        ${h.sst > 24 && h.sst < 30 ? '✅' : '⚠️'}</span>` : '--';
        document.getElementById('detail-chl').textContent =
            h.chl ? `${h.chl.toFixed(3)} mg/m³` : '--';
        document.getElementById('detail-do').textContent =
            h.do_ml ? `${h.do_ml.toFixed(1)} ml/L` : '--';
        document.getElementById('detail-ssh').textContent =
            h.ssh != null ? `${h.ssh.toFixed(3)}m` : '--';

        // 代謝指數
        const phi = h.phi || h.phi_viability;
        document.getElementById('detail-phi').textContent =
            phi ? `Φ = ${phi.toFixed(1)}` : 'Φ = --';
        document.getElementById('detail-phi').style.color =
            phi > 3 ? 'var(--accent-green)' : phi > 2 ? 'var(--accent-yellow)' : 'var(--accent-red)';

        // 載入食物鏈詳情
        loadFoodChainDetail(h);

        // 聚焦地圖
        map.flyTo([h.lat || 22, h.lon || 140], 7, { duration: 1 });
    }

    async function loadFoodChainDetail(h) {
        try {
            const data = await apiFetch(
                `/api/v1/food_chain_timeline?lat=${h.lat}&lon=${h.lon}&species=${currentSpecies}`
            );

            document.getElementById('fc-phyto').textContent =
                `${data.bloom?.phase || '--'} (${(data.bloom?.intensity * 100 || 0).toFixed(0)}%)`;
            document.getElementById('fc-zoo').textContent =
                `${data.zooplankton?.status || '--'} (${(data.zooplankton?.biomass_index * 100 || 0).toFixed(0)}%)`;
            document.getElementById('fc-bait').textContent =
                data.current_stage_zh || '--';
            document.getElementById('fc-tuna').textContent =
                data.tuna_arrival?.optimal_date || '--';

            // DVM & 月相
            document.getElementById('detail-dvm').textContent =
                `日: --m / 夜: --m`;

            // 進食窗口
            const feedRes = await apiFetch(
                `/api/v1/feeding_windows?lat=${h.lat}&lon=${h.lon}&species=${currentSpecies}`
            ).catch(() => null);

            if (feedRes?.feeding_windows?.length) {
                const best = feedRes.feeding_windows[0];
                document.getElementById('detail-feeding-window').textContent =
                    `${best.start}-${best.end}`;
            }

            // 月相
            document.getElementById('detail-moon').textContent =
                feedRes?.lunar_affected ? '🌑 新月 (表層覓食)' : '--';

        } catch (err) {
            console.warn('Food chain detail error:', err);
        }
    }

    // ═══ QUICK STATS ═══
    function renderQuickStats(hs, hotspotsRes, seaCondRes) {
        const count = hs.length;
        const maxHSI = hs.length ? Math.max(...hs.map(h => (h.score || h.hsi || 0))) : 0;
        const recommended = hs.filter(h => (h.score || h.hsi || 0) > 0.6).length;

        document.getElementById('stat-hotspot-count').textContent = count;
        document.getElementById('stat-max-hsi').textContent = `${Math.round(maxHSI * 100)}%`;
        document.getElementById('stat-recommend').textContent = `${recommended} 區`;

        if (seaCondRes) {
            const sstMin = seaCondRes.sst_min;
            const sstMax = seaCondRes.sst_max;
            document.getElementById('stat-sst-range').textContent =
                sstMin && sstMax ? `${sstMin}°C - ${sstMax}°C` : '--';
            document.getElementById('stat-wind').textContent =
                seaCondRes.wind_speed ? `${seaCondRes.wind_speed} kn` : '--';
        }
    }

    function renderStarRatings(hs) {
        const avgHSI = hs.length ?
            hs.reduce((s, h) => s + (h.score || h.hsi || 0), 0) / hs.length : 0;

        // 基於平均 HSI 生成星級 (簡化)
        renderStars('star-today', Math.round(avgHSI * 5));
        renderStars('star-tomorrow', Math.round(avgHSI * 5) + 1);
        renderStars('star-day3', Math.min(5, Math.round(avgHSI * 5) + 1));
    }

    function renderStars(elementId, count) {
        const el = document.getElementById(elementId);
        count = Math.max(1, Math.min(5, count));
        el.innerHTML = '';
        for (let i = 0; i < 5; i++) {
            const span = document.createElement('span');
            span.className = i < count ? 'star-filled' : 'star-empty';
            span.textContent = '⭐';
            el.appendChild(span);
        }
    }

    // ═══ FOOD CHAIN TIMELINE (D3) ═══
    async function loadFoodChainTimeline() {
        try {
            const firstHotspot = hotspots[0];
            if (!firstHotspot) return;

            const data = await apiFetch(
                `/api/v1/food_chain_timeline?lat=${firstHotspot.lat}&lon=${firstHotspot.lon}&species=${currentSpecies}`
            );
            timelineData = data;

            // 更新食物鏈指數
            const fcScore = data.food_chain_score || 0;
            document.getElementById('fc-progress').style.width = `${fcScore * 100}%`;
            document.getElementById('fc-score-text').textContent =
                `${(fcScore * 100).toFixed(0)}% — ${data.current_stage_zh || ''}`;

            // 更新時間線物種標籤
            document.getElementById('timeline-species').textContent =
                SPECIES_ZH[currentSpecies] || currentSpecies;

            renderTimeline(data);
        } catch (err) {
            console.warn('Timeline load error:', err);
        }
    }

    function renderTimeline(data) {
        const svg = d3.select('#timeline-canvas');
        svg.selectAll('*').remove();

        const container = document.getElementById('timeline-canvas');
        const width = container.clientWidth || 800;
        const height = container.clientHeight || 150;

        svg.attr('width', width).attr('height', height);

        // 時間軸: -7d → +8d
        const margin = { top: 20, right: 30, bottom: 30, left: 30 };
        const iw = width - margin.left - margin.right;
        const ih = height - margin.top - margin.bottom;

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleLinear().domain([-7, 8]).range([0, iw]);

        // 背景格線
        for (let d = -7; d <= 8; d++) {
            g.append('line')
                .attr('x1', x(d)).attr('x2', x(d))
                .attr('y1', 0).attr('y2', ih)
                .attr('stroke', '#2d3748').attr('stroke-width', 0.5);
        }

        // 日期標籤
        for (let d = -7; d <= 8; d += 2) {
            g.append('text')
                .attr('x', x(d)).attr('y', ih + 18)
                .attr('text-anchor', 'middle').attr('fill', '#64748b').attr('font-size', 10)
                .text(d === 0 ? '今日' : (d > 0 ? `+${d}d` : `${d}d`));
        }

        // 今日現在線
        g.append('line')
            .attr('x1', x(0)).attr('x2', x(0))
            .attr('y1', -5).attr('y2', ih + 5)
            .attr('stroke', '#f87171').attr('stroke-width', 2)
            .attr('stroke-dasharray', '4,2');

        g.append('text')
            .attr('x', x(0)).attr('y', -8)
            .attr('text-anchor', 'middle').attr('fill', '#f87171').attr('font-size', 10)
            .text('NOW');

        // 營養級條帶
        const levels = [
            { name: '🌿 浮游植物', color: '#34d399', y: 0, start: -5, end: 0 },
            { name: '🦐 浮游動物', color: '#22d3ee', y: 1, start: -2, end: 4 },
            { name: '🐠 餌料魚', color: '#fbbf24', y: 2, start: 1, end: 6 },
            { name: '🐟 鮪魚到達', color: '#f87171', y: 3, start: 3, end: 8 },
        ];

        // 根據實際數據調整
        if (data.bloom) {
            const bloomDays = data.bloom.phase === 'peak' ? 0 :
                data.bloom.phase === 'decline' ? -3 : -5;
            levels[0].start = bloomDays - 3;
            levels[0].end = bloomDays + 3;
        }

        if (data.zooplankton) {
            const zooETA = data.zooplankton.peak_eta_days || 5;
            levels[1].start = levels[0].end - 2;
            levels[1].end = levels[0].end + zooETA + 3;
        }

        if (data.tuna_arrival?.arrival_window) {
            const [arrMin, arrMax] = data.tuna_arrival.arrival_window;
            levels[3].start = arrMin;
            levels[3].end = arrMax;
        }

        const barH = (ih - 10) / 4 - 4;

        levels.forEach((lv, i) => {
            const yPos = i * (barH + 4);

            // 條帶
            g.append('rect')
                .attr('x', x(Math.max(lv.start, -7)))
                .attr('y', yPos)
                .attr('width', Math.max(0, x(Math.min(lv.end, 8)) - x(Math.max(lv.start, -7))))
                .attr('height', barH)
                .attr('rx', 4)
                .attr('fill', lv.color)
                .attr('opacity', 0.3);

            // 進度 (到今日)
            const progressEnd = Math.min(0, lv.end);
            if (progressEnd > lv.start) {
                g.append('rect')
                    .attr('x', x(Math.max(lv.start, -7)))
                    .attr('y', yPos)
                    .attr('width', Math.max(0, x(progressEnd) - x(Math.max(lv.start, -7))))
                    .attr('height', barH)
                    .attr('rx', 4)
                    .attr('fill', lv.color)
                    .attr('opacity', 0.7);
            }

            // 標籤
            g.append('text')
                .attr('x', x(Math.max(lv.start, -7)) + 6)
                .attr('y', yPos + barH / 2 + 4)
                .attr('fill', '#fff').attr('font-size', 11).attr('font-weight', 500)
                .text(lv.name);
        });

        // 最佳漁獲窗口高亮
        if (data.tuna_arrival?.arrival_window) {
            const [arrMin, arrMax] = data.tuna_arrival.arrival_window;
            g.append('rect')
                .attr('x', x(arrMin))
                .attr('y', -2)
                .attr('width', x(arrMax) - x(arrMin))
                .attr('height', ih + 4)
                .attr('rx', 4)
                .attr('fill', '#fbbf24')
                .attr('opacity', 0.08)
                .attr('stroke', '#fbbf24')
                .attr('stroke-width', 1)
                .attr('stroke-dasharray', '4,2');
        }
    }

    // ═══ MOON PHASE ═══
    async function loadMoonPhase() {
        try {
            const data = await apiFetch('/api/v1/food_chain?lat=25&lon=130&species=' + currentSpecies);
            if (data.lunar_phase) {
                document.getElementById('stat-moon-name').textContent = data.lunar_phase;
                // 簡單映射 emoji
                const phaseMap = {
                    '新月': '🌑', '眉月': '🌒', '上弦': '🌓', '盈凸': '🌔',
                    '滿月': '🌕', '虧凸': '🌖', '下弦': '🌗', '殘月': '🌘'
                };
                document.getElementById('stat-moon-emoji').textContent =
                    phaseMap[data.lunar_phase] || '🌙';
            }
        } catch (err) {
            console.warn('Moon phase error:', err);
        }
    }

    // ═══ SHAP MODAL ═══
    async function showSHAP() {
        if (!selectedHotspot) return;

        const modal = document.getElementById('shap-modal');
        const body = document.getElementById('shap-body');
        modal.classList.add('active');

        try {
            const data = await apiFetch(`/api/v1/explain?hotspot_id=${selectedHotspot.index}`);
            const factors = data.shap_factors || [];

            body.innerHTML = `
            <div style="margin-bottom:16px;font-size:13px;color:var(--text-secondary)">
                HSI: ${Math.round((data.hsi || 0) * 100)}% | ${SPECIES_ZH[data.species] || data.species}
            </div>
            ${factors.map(f => {
                const pct = Math.abs(f.contribution) * 100;
                const cls = f.contribution >= 0 ? 'positive' : 'negative';
                return `
                    <div class="shap-bar-row">
                        <span class="shap-label">${f.label}</span>
                        <div class="shap-bar-container">
                            <div class="shap-bar ${cls}" style="width:${Math.min(pct, 100)}%"></div>
                        </div>
                        <span class="shap-value">${f.contribution >= 0 ? '+' : ''}${f.contribution.toFixed(2)}</span>
                    </div>
                `;
            }).join('')}
            <div style="margin-top:16px;font-size:11px;color:var(--text-muted)">
                方法: ${data.shap_method || 'weight-based'}
            </div>
        `;
        } catch (err) {
            body.innerHTML = `<div style="color:var(--accent-red)">載入失敗: ${err.message}</div>`;
        }
    }

    // ═══ EVENTS ═══
    function bindEvents() {
        // 物種選擇
        document.getElementById('species-select').addEventListener('change', e => {
            currentSpecies = e.target.value;
            loadFoodChainTimeline();
        });

        // 回列表
        document.getElementById('btn-back-list').addEventListener('click', () => {
            document.getElementById('hotspot-list').style.display = '';
            document.getElementById('detail-panel').style.display = 'none';
            selectedHotspot = null;
        });

        // 刷新
        document.getElementById('btn-refresh').addEventListener('click', loadData);

        // SHAP
        document.getElementById('btn-shap').addEventListener('click', showSHAP);
        document.getElementById('shap-close').addEventListener('click', () => {
            document.getElementById('shap-modal').classList.remove('active');
        });
        document.getElementById('shap-modal').addEventListener('click', e => {
            if (e.target.id === 'shap-modal') {
                e.target.classList.remove('active');
            }
        });

        // 圖層切換
        document.getElementById('layer-heatmap').addEventListener('change', () => renderHeatmap(hotspots));
        document.getElementById('layer-hotspots').addEventListener('change', () => renderHotspotMarkers(hotspots));

        // [v13] 黑潮圖層切換
        const kuroshioToggle = document.getElementById('layer-kuroshio');
        if (kuroshioToggle) {
            kuroshioToggle.addEventListener('change', () => {
                if (kuroshioToggle.checked && kuroshioLayer) {
                    kuroshioLayer.addTo(map);
                } else if (kuroshioLayer) {
                    map.removeLayer(kuroshioLayer);
                }
            });
        }

        // 時間滑桿
        document.getElementById('time-slider').addEventListener('input', e => {
            const hours = parseInt(e.target.value);
            if (hours === 0) {
                document.getElementById('time-label').textContent = '現在';
            } else if (hours < 0) {
                document.getElementById('time-label').textContent = `${hours}h (歷史)`;
            } else {
                const days = Math.floor(hours / 24);
                const h = hours % 24;
                document.getElementById('time-label').textContent =
                    days > 0 ? `+${days}d ${h}h (預報)` : `+${h}h (預報)`;
            }
        });

        // 播放按鈕
        let playInterval = null;
        document.getElementById('btn-play').addEventListener('click', function () {
            if (playInterval) {
                clearInterval(playInterval);
                playInterval = null;
                this.textContent = '▶';
                return;
            }
            this.textContent = '⏸';
            const slider = document.getElementById('time-slider');
            slider.value = -24;
            playInterval = setInterval(() => {
                let v = parseInt(slider.value) + 3;
                if (v > 192) {
                    v = -24;
                    clearInterval(playInterval);
                    playInterval = null;
                    document.getElementById('btn-play').textContent = '▶';
                }
                slider.value = v;
                slider.dispatchEvent(new Event('input'));
            }, 200);
        });

        // 搜尋
        document.getElementById('search-input').addEventListener('keydown', e => {
            if (e.key === 'Enter') {
                const val = e.target.value.trim();
                const match = val.match(/(-?[\d.]+)\s*[,，]\s*(-?[\d.]+)/);
                if (match) {
                    const lat = parseFloat(match[1]);
                    const lon = parseFloat(match[2]);
                    map.flyTo([lat, lon], 8, { duration: 1 });
                }
            }
        });

        // Mobile toggles
        document.getElementById('btn-sidebar-toggle')?.addEventListener('click', () => {
            document.getElementById('left-sidebar').classList.toggle('mobile-open');
            document.getElementById('right-panel').classList.remove('mobile-open');
        });
        document.getElementById('btn-stats-toggle')?.addEventListener('click', () => {
            document.getElementById('right-panel').classList.toggle('mobile-open');
            document.getElementById('left-sidebar').classList.remove('mobile-open');
        });

        // 視窗大小變化 → 重繪時間線
        window.addEventListener('resize', () => {
            if (timelineData) renderTimeline(timelineData);
        });
    }

    // ═══ Auto Refresh (每 5 分鐘) ═══
    function startAutoRefresh() {
        setInterval(loadData, 5 * 60 * 1000);
    }

})();
