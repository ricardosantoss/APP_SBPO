// service-worker.js
self.addEventListener('fetch', function(event) {
  // Simple network-first fetch strategy
  event.respondWith(
    fetch(event.request).catch(function() {
      // Offline fallback can be added here if needed
    })
  );
});
