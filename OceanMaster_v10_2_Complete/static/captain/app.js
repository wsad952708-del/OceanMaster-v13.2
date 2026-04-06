/**
 * OceanMaster Captain — Offline-First App Logic
 * IndexedDB 離線佇列 + navigator.onLine 自動同步
 */

const API_BASE = '/api/v1';
const DB_NAME = 'CaptainReports';
const STORE_NAME = 'pending_reports';
const DB_VERSION = 1;

let db = null;

// ── IndexedDB 初始化 ──
function openDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = (e) => {
            const _db = e.target.result;
            if (!_db.objectStoreNames.contains(STORE_NAME)) {
                _db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
            }
        };
        req.onsuccess = (e) => { db = e.target.result; resolve(db); };
        req.onerror = (e) => reject(e.target.error);
    });
}

// ── 新增 pending 回報到 IndexedDB ──
function addPending(report) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        store.add(report);
        tx.oncomplete = () => resolve();
        tx.onerror = (e) => reject(e.target.error);
    });
}

// ── 取得所有 pending 回報 ──
function getAllPending() {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const req = store.getAll();
        req.onsuccess = () => resolve(req.result);
        req.onerror = (e) => reject(e.target.error);
    });
}

// ── 刪除已同步的回報 ──
function deletePending(id) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        store.delete(id);
        tx.oncomplete = () => resolve();
        tx.onerror = (e) => reject(e.target.error);
    });
}

// ── 送出回報 (online → API, offline → IndexedDB) ──
async function submitReport() {
    const vesselId = document.getElementById('vesselId').value.trim();
    const species = document.getElementById('species').value;
    const cpue = parseFloat(document.getElementById('cpue').value);
    const lat = parseFloat(document.getElementById('lat').value);
    const lon = parseFloat(document.getElementById('lon').value);

    // Validate
    if (!vesselId) { showToast('請輸入船舶編號', 'warning'); return; }
    if (isNaN(cpue) || cpue < 0) { showToast('請輸入有效捕獲量', 'warning'); return; }
    if (isNaN(lat) || lat < -90 || lat > 90) { showToast('緯度範圍: -90~90', 'warning'); return; }
    if (isNaN(lon) || lon < -180 || lon > 360) { showToast('經度範圍: -180~360', 'warning'); return; }

    const report = {
        vessel_id: vesselId,
        timestamp: new Date().toISOString(),
        lat, lon,
        target_species: species,
        actual_cpue_kg_day: cpue,
        env_snapshot: { submitted_via: 'captain_pwa' },
        status: 'pending',
    };

    if (navigator.onLine) {
        // 直接送
        try {
            const resp = await fetch(`${API_BASE}/report_catch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': 'captain' },
                body: JSON.stringify(report),
            });
            if (resp.ok) {
                report.status = 'synced';
                showToast('✅ 回報已送出!', 'success');
            } else {
                // API 錯誤 → 存到 IndexedDB
                report.status = 'pending';
                await addPending(report);
                showToast('⚠️ 伺服器錯誤，已暫存', 'warning');
            }
        } catch (e) {
            report.status = 'pending';
            await addPending(report);
            showToast('⚠️ 網路異常，已暫存', 'warning');
        }
    } else {
        // 離線 → IndexedDB
        await addPending(report);
        showToast('📱 離線暫存成功 (上線後自動同步)', 'warning');
    }

    // 存到歷史 (localStorage)
    saveHistory(report);
    renderHistory();
    updateQueueCount();

    // 清空表單 (保留船號)
    document.getElementById('cpue').value = '';
}

// ── 自動同步佇列 ──
async function syncPendingReports() {
    if (!navigator.onLine || !db) return;

    const pending = await getAllPending();
    if (pending.length === 0) return;

    console.log(`[Sync] ${pending.length} reports to sync`);
    let synced = 0;

    for (const report of pending) {
        try {
            const resp = await fetch(`${API_BASE}/report_catch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': 'captain' },
                body: JSON.stringify(report),
            });
            if (resp.ok) {
                await deletePending(report.id);
                synced++;
                // 更新歷史狀態
                updateHistoryStatus(report.timestamp, 'synced');
            }
        } catch (e) {
            console.warn(`[Sync] Failed: ${e.message}`);
            break; // 停止, 下次再試
        }
    }

    if (synced > 0) {
        showToast(`✅ 已同步 ${synced} 筆離線回報`, 'success');
        renderHistory();
    }
    updateQueueCount();
}

// ── 網路狀態偵測 ──
function updateOnlineStatus() {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    if (navigator.onLine) {
        dot.classList.add('online');
        text.textContent = '線上';
        // 上線 → 立即同步
        setTimeout(syncPendingReports, 1000);
    } else {
        dot.classList.remove('online');
        text.textContent = '離線 (資料暫存中)';
    }
}

// ── Toast ──
function showToast(msg, type) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = `toast ${type} visible`;
    setTimeout(() => { t.classList.remove('visible'); }, 3000);
}

// ── Queue bar ──
async function updateQueueCount() {
    if (!db) return;
    const pending = await getAllPending();
    const bar = document.getElementById('queueBar');
    const count = document.getElementById('queueCount');
    if (pending.length > 0) {
        count.textContent = pending.length;
        bar.classList.add('visible');
    } else {
        bar.classList.remove('visible');
    }
}

// ── 歷史記錄 (localStorage) ──
function saveHistory(report) {
    const h = JSON.parse(localStorage.getItem('captain_history') || '[]');
    h.unshift(report);
    if (h.length > 20) h.length = 20;
    localStorage.setItem('captain_history', JSON.stringify(h));
}

function updateHistoryStatus(timestamp, status) {
    const h = JSON.parse(localStorage.getItem('captain_history') || '[]');
    for (const item of h) {
        if (item.timestamp === timestamp) {
            item.status = status;
            break;
        }
    }
    localStorage.setItem('captain_history', JSON.stringify(h));
}

function renderHistory() {
    const list = document.getElementById('historyList');
    const h = JSON.parse(localStorage.getItem('captain_history') || '[]');
    list.innerHTML = h.slice(0, 5).map(r => `
        <div class="history-item">
            <div>
                <strong>${r.target_species}</strong> ${r.actual_cpue_kg_day}kg/d<br>
                <small>${r.lat?.toFixed(2)}°N, ${r.lon?.toFixed(2)}°E</small>
            </div>
            <span class="badge ${r.status}">${r.status === 'synced' ? '✅ 已同步' : '⏳ 待同步'}</span>
        </div>
    `).join('');
}

// ── 自動定時同步 (每 30 秒) ──
setInterval(syncPendingReports, 30000);

// ── Init ──
window.addEventListener('online', updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);

(async () => {
    await openDB();
    updateOnlineStatus();
    updateQueueCount();
    renderHistory();

    // 自動填入上次船號
    const lastVessel = localStorage.getItem('last_vessel_id');
    if (lastVessel) document.getElementById('vesselId').value = lastVessel;

    // 保存船號
    document.getElementById('vesselId').addEventListener('change', (e) => {
        localStorage.setItem('last_vessel_id', e.target.value);
    });

    // 嘗試 GPS 自動填入
    if ('geolocation' in navigator) {
        navigator.geolocation.getCurrentPosition((pos) => {
            const latInput = document.getElementById('lat');
            const lonInput = document.getElementById('lon');
            if (!latInput.value) latInput.value = pos.coords.latitude.toFixed(4);
            if (!lonInput.value) lonInput.value = pos.coords.longitude.toFixed(4);
        }, () => { }, { timeout: 5000 });
    }
})();
