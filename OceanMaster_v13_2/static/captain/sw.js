/**
 * OceanMaster Captain — Service Worker
 * Offline cache + background sync
 */
const CACHE_NAME = 'captain-v1';
const CACHED_URLS = [
    '/captain/',
    '/captain/index.html',
    '/captain/app.js',
    '/captain/manifest.json',
];

// Install — 預快取核心資源
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(CACHED_URLS);
        })
    );
    self.skipWaiting();
});

// Activate — 清除舊快取
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((names) => {
            return Promise.all(
                names.filter(n => n !== CACHE_NAME).map(n => caches.delete(n))
            );
        })
    );
    self.clients.claim();
});

// Fetch — Cache-first for static, network-first for API
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // API 請求 → 網路優先
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(
            fetch(event.request).catch(() => {
                return new Response(
                    JSON.stringify({ error: 'offline', status: 'queued' }),
                    { headers: { 'Content-Type': 'application/json' } }
                );
            })
        );
        return;
    }

    // 靜態資源 → 快取優先
    event.respondWith(
        caches.match(event.request).then((cached) => {
            return cached || fetch(event.request).then((resp) => {
                // 快取新資源
                const clone = resp.clone();
                caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
                return resp;
            }).catch(() => {
                // 完全離線 fallback
                if (event.request.destination === 'document') {
                    return caches.match('/captain/index.html');
                }
            });
        })
    );
});
