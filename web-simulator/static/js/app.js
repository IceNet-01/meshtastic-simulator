/**
 * Meshtastic Web Simulator - Frontend Application
 * With OpenStreetMap background and drag-to-reposition nodes
 */

// ============== State Management ==============
const state = {
    nodes: new Map(),
    links: [],  // RF-calculated links from server
    selectedNodes: new Set(),
    config: null,
    socket: null,
    canvas: null,
    ctx: null,

    // Map state - using lat/lon as center
    mapCenter: { lat: 39.8283, lon: -98.5795 }, // Default: center of USA
    zoomLevel: 12, // OSM zoom level (1-19)

    // Canvas interaction state
    viewOffset: { x: 0, y: 0 },
    isDragging: false,
    dragMode: 'pan', // 'pan' or 'node'
    draggedNode: null,
    lastMouse: { x: 0, y: 0 },

    // Interaction state
    editingNode: null,
    highlightedRoute: null,

    // Map tiles
    tileCache: new Map(),
    tileSize: 256,
    loadingTiles: new Set(),

    // Animation state
    animation: {
        active: false,
        type: null,  // 'broadcast', 'dm', 'traceroute'
        data: null,
        startTime: 0,
        packets: [],  // Moving packet animations
        nodeStates: new Map(),  // Node visual states during animation
        currentHop: 0
    }
};

// ============== Constants ==============
const TILE_SERVER = 'https://tile.openstreetmap.org';
const METERS_PER_DEGREE_LAT = 111320; // Approximate meters per degree latitude

// Render loop state
let renderLoopRunning = false;
let lastRenderTime = 0;
let animationFallbackInterval = null;

/**
 * Start a fallback interval that ensures animations continue
 * even when requestAnimationFrame is throttled by the browser.
 */
function startAnimationFallback() {
    if (animationFallbackInterval) {
        clearInterval(animationFallbackInterval);
    }
    animationFallbackInterval = setInterval(() => {
        if (state.animation.active) {
            startRenderLoop();
        } else {
            clearInterval(animationFallbackInterval);
            animationFallbackInterval = null;
        }
    }, 100); // Check every 100ms
}

/**
 * Restart the render loop for active animations.
 * Called when tab becomes visible or window gains focus to ensure
 * animations continue smoothly after being throttled.
 */
function startRenderLoop() {
    if (!state.animation.active) {
        renderLoopRunning = false;
        return;
    }

    // Force a render immediately to update the display
    render();

    renderLoopRunning = true;
    lastRenderTime = performance.now();

    // Restart the appropriate animation based on type
    switch (state.animation.type) {
        case 'broadcast':
            requestAnimationFrame(animateBroadcast);
            break;
        case 'dm':
            // Check if using flood or path-based animation
            if (state.animation.data.propagation && state.animation.data.propagation.length > 0) {
                requestAnimationFrame(animateDMFlood);
            } else {
                requestAnimationFrame(animateDM);
            }
            break;
        case 'traceroute':
            // Check if using flood or path-based animation
            if (state.animation.data.propagation && state.animation.data.propagation.length > 0) {
                requestAnimationFrame(animateTracerouteFlood);
            } else {
                requestAnimationFrame(animateTraceroute);
            }
            break;
        default:
            renderLoopRunning = false;
    }
}

// Debug mode - set to true to enable console logging
const DEBUG = false;
function debugLog(...args) {
    if (DEBUG) console.log(...args);
}

// ============== Initialization ==============
document.addEventListener('DOMContentLoaded', () => {
    debugLog('Meshtastic Web Simulator initializing...');

    try {
        initCanvas();
    } catch (e) {
        console.error('Canvas init failed:', e);
    }

    try {
        initSocket();
    } catch (e) {
        console.error('Socket init failed:', e);
    }

    try {
        initEventListeners();
    } catch (e) {
        console.error('Event listeners init failed:', e);
    }

    loadConfig();
    loadNodes();

    // Try to get user's location for initial map center
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                state.mapCenter.lat = pos.coords.latitude;
                state.mapCenter.lon = pos.coords.longitude;
                render();
                log('Map centered on your location', 'info');
            },
            () => {
                log('Using default map location (center of USA)', 'info');
                render();
            },
            { timeout: 5000 }
        );
    } else {
        render();
    }

    // Handle visibility changes - restart render loop when tab becomes visible
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && state.animation.active) {
            startRenderLoop();
        }
    });

    // Also handle window focus
    window.addEventListener('focus', () => {
        if (state.animation.active) {
            startRenderLoop();
        }
    });

    debugLog('Initialization complete');
});

function initCanvas() {
    state.canvas = document.getElementById('network-canvas');
    state.ctx = state.canvas.getContext('2d');

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Canvas events
    state.canvas.addEventListener('click', handleCanvasClick);
    state.canvas.addEventListener('dblclick', handleCanvasDoubleClick);
    state.canvas.addEventListener('mousedown', handleMouseDown);
    state.canvas.addEventListener('mousemove', handleMouseMove);
    state.canvas.addEventListener('mouseup', handleMouseUp);
    state.canvas.addEventListener('wheel', handleWheel);
    state.canvas.addEventListener('mouseleave', handleMouseLeave);

    // Touch events for mobile
    state.canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    state.canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    state.canvas.addEventListener('touchend', handleTouchEnd);
}

function resizeCanvas() {
    const container = state.canvas.parentElement;
    state.canvas.width = container.clientWidth;
    state.canvas.height = container.clientHeight - 60; // toolbar + status bar
    render();
}

function initSocket() {
    state.socket = io();

    state.socket.on('connect', () => {
        log('Connected to server', 'info');
    });

    state.socket.on('disconnect', () => {
        log('Disconnected from server', 'warning');
    });

    state.socket.on('node_added', (node) => {
        state.nodes.set(node.id, node);
        updateNodeList();
        updateCommandSelects();
        render();
        log(`Node ${node.id} (${node.name}) added`, 'success');
    });

    state.socket.on('node_updated', (node) => {
        state.nodes.set(node.id, node);
        updateNodeList();
        render();
    });

    state.socket.on('node_removed', (data) => {
        state.nodes.delete(data.id);
        state.selectedNodes.delete(data.id);
        updateNodeList();
        updateCommandSelects();
        render();
        log(`Node ${data.id} removed`, 'info');
    });

    state.socket.on('nodes_cleared', () => {
        state.nodes.clear();
        state.selectedNodes.clear();
        updateNodeList();
        updateCommandSelects();
        render();
        log('All nodes cleared', 'info');
    });

    state.socket.on('config_imported', (data) => {
        loadNodes();
        log(`Imported ${data.nodeCount} nodes`, 'success');
    });

    state.socket.on('command_response', (data) => {
        log(data.message, data.status === 'error' || data.status === 'failed' ? 'error' : 'success');

        // Handle simulation visualization
        if (data.simulation) {
            if (data.command === 'broadcast') {
                startBroadcastAnimation(data.simulation);
            } else if (data.command === 'dm') {
                startDMAnimation(data.simulation);
            } else if (data.command === 'traceroute') {
                startTracerouteAnimation(data.simulation);
            }
        } else if (data.route) {
            // Legacy route highlight
            state.highlightedRoute = data.route;
            render();
            setTimeout(() => {
                state.highlightedRoute = null;
                render();
            }, 5000);
        }
    });

    state.socket.on('simulation_status', (data) => {
        log(`Simulation: ${data.message}`, data.status === 'error' ? 'error' : 'info');
    });
}

function initEventListeners() {
    // Add node button
    document.getElementById('btn-add-node').addEventListener('click', addNodeFromForm);

    // Hop limit slider
    document.getElementById('node-hoplimit').addEventListener('input', (e) => {
        document.getElementById('node-hoplimit-value').textContent = e.target.value;
    });
    document.getElementById('edit-hoplimit').addEventListener('input', (e) => {
        document.getElementById('edit-hoplimit-value').textContent = e.target.value;
    });

    // Zoom controls
    document.getElementById('btn-zoom-in').addEventListener('click', () => {
        state.zoomLevel = Math.min(state.zoomLevel + 1, 19);
        updateZoomDisplay();
        render();
    });
    document.getElementById('btn-zoom-out').addEventListener('click', () => {
        state.zoomLevel = Math.max(state.zoomLevel - 1, 1);
        updateZoomDisplay();
        render();
    });
    document.getElementById('btn-fit').addEventListener('click', fitAllNodes);

    // Clear all
    document.getElementById('btn-clear').addEventListener('click', () => {
        if (confirm('Clear all nodes?')) {
            fetch('/api/nodes/clear', { method: 'POST' });
        }
    });

    // Export/Import
    document.getElementById('btn-export').addEventListener('click', exportYaml);
    document.getElementById('btn-import').addEventListener('click', () => {
        document.getElementById('import-modal').classList.remove('hidden');
    });
    document.getElementById('btn-import-cancel').addEventListener('click', () => {
        document.getElementById('import-modal').classList.add('hidden');
    });
    document.getElementById('btn-import-confirm').addEventListener('click', importYaml);

    // NodeDB Import
    document.getElementById('btn-import-nodedb').addEventListener('click', () => {
        document.getElementById('nodedb-modal').classList.remove('hidden');
        document.getElementById('nodedb-preview').classList.add('hidden');
    });
    document.getElementById('btn-nodedb-cancel').addEventListener('click', () => {
        document.getElementById('nodedb-modal').classList.add('hidden');
        document.getElementById('nodedb-json').value = '';
        document.getElementById('nodedb-preview').classList.add('hidden');
    });
    document.getElementById('btn-nodedb-preview').addEventListener('click', previewNodeDB);
    document.getElementById('btn-nodedb-import').addEventListener('click', importNodeDB);

    // Edit modal
    document.getElementById('btn-edit-cancel').addEventListener('click', closeEditModal);
    document.getElementById('btn-edit-save').addEventListener('click', saveEditedNode);
    document.getElementById('btn-edit-delete').addEventListener('click', deleteEditedNode);

    // Commands
    document.getElementById('btn-broadcast').addEventListener('click', sendBroadcast);
    document.getElementById('btn-dm').addEventListener('click', sendDM);
    document.getElementById('btn-traceroute').addEventListener('click', sendTraceroute);

    // Settings
    document.getElementById('btn-apply-settings').addEventListener('click', applySettings);

    // Clear log
    document.getElementById('btn-clear-log').addEventListener('click', () => {
        document.getElementById('message-log').innerHTML = '';
    });

    // Statistics refresh
    document.getElementById('btn-refresh-stats').addEventListener('click', refreshStatistics);

    // GPS coordinate toggle
    document.getElementById('use-gps-coords').addEventListener('change', (e) => {
        document.getElementById('coord-xy').classList.toggle('hidden', e.target.checked);
        document.getElementById('coord-gps').classList.toggle('hidden', !e.target.checked);
    });

    // Channel configuration
    document.getElementById('channel-psk-type').addEventListener('change', (e) => {
        const customInput = document.getElementById('channel-psk-custom');
        customInput.classList.toggle('hidden', e.target.value !== 'custom');
        updateEncryptionStatus();
    });
    document.getElementById('channel-encryption-enabled').addEventListener('change', updateEncryptionStatus);

    // Save/Load Scenario
    document.getElementById('btn-save-scenario').addEventListener('click', () => {
        document.getElementById('save-modal').classList.remove('hidden');
    });
    document.getElementById('btn-save-cancel').addEventListener('click', () => {
        document.getElementById('save-modal').classList.add('hidden');
    });
    document.getElementById('btn-save-confirm').addEventListener('click', saveScenario);

    document.getElementById('btn-load-scenario').addEventListener('click', () => {
        document.getElementById('load-modal').classList.remove('hidden');
    });
    document.getElementById('btn-load-cancel').addEventListener('click', () => {
        document.getElementById('load-modal').classList.add('hidden');
        document.getElementById('scenario-preview').classList.add('hidden');
        document.getElementById('scenario-file').value = '';
    });
    document.getElementById('scenario-file').addEventListener('change', previewScenarioFile);
    document.getElementById('btn-load-confirm').addEventListener('click', loadScenario);
}

// ============== Geo Coordinate Conversions ==============
function metersPerPixel(lat, zoom) {
    // Calculate meters per pixel at given latitude and zoom level
    return 156543.03392 * Math.cos(lat * Math.PI / 180) / Math.pow(2, zoom);
}

function latLonToPixel(lat, lon, zoom) {
    // Convert lat/lon to pixel coordinates (Web Mercator projection)
    const scale = Math.pow(2, zoom) * state.tileSize;
    const x = (lon + 180) / 360 * scale;
    const latRad = lat * Math.PI / 180;
    const y = (1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * scale;
    return { x, y };
}

function pixelToLatLon(px, py, zoom) {
    // Convert pixel coordinates back to lat/lon
    const scale = Math.pow(2, zoom) * state.tileSize;
    const lon = px / scale * 360 - 180;
    const latRad = Math.atan(Math.sinh(Math.PI * (1 - 2 * py / scale)));
    const lat = latRad * 180 / Math.PI;
    return { lat, lon };
}

function metersToLatLon(x, y, centerLat, centerLon) {
    // Convert local meter coordinates to lat/lon
    // x = east/west (positive = east), y = north/south (positive = north)
    const lat = centerLat + y / METERS_PER_DEGREE_LAT;
    const metersPerDegreeLon = METERS_PER_DEGREE_LAT * Math.cos(centerLat * Math.PI / 180);
    const lon = centerLon + x / metersPerDegreeLon;
    return { lat, lon };
}

function latLonToMeters(lat, lon, centerLat, centerLon) {
    // Convert lat/lon to local meter coordinates relative to center
    const y = (lat - centerLat) * METERS_PER_DEGREE_LAT;
    const metersPerDegreeLon = METERS_PER_DEGREE_LAT * Math.cos(centerLat * Math.PI / 180);
    const x = (lon - centerLon) * metersPerDegreeLon;
    return { x, y };
}

function worldToScreen(x, y) {
    // Convert world coordinates (meters relative to map center) to screen pixels
    const mpp = metersPerPixel(state.mapCenter.lat, state.zoomLevel);
    return {
        x: state.canvas.width / 2 + x / mpp + state.viewOffset.x,
        y: state.canvas.height / 2 - y / mpp + state.viewOffset.y
    };
}

function screenToWorld(sx, sy) {
    // Convert screen pixels to world coordinates (meters relative to map center)
    const mpp = metersPerPixel(state.mapCenter.lat, state.zoomLevel);
    return {
        x: (sx - state.canvas.width / 2 - state.viewOffset.x) * mpp,
        y: -(sy - state.canvas.height / 2 - state.viewOffset.y) * mpp
    };
}

function screenToLatLon(sx, sy) {
    const world = screenToWorld(sx, sy);
    return metersToLatLon(world.x, world.y, state.mapCenter.lat, state.mapCenter.lon);
}

// ============== Map Tile Loading ==============
function getTileUrl(x, y, z) {
    return `${TILE_SERVER}/${z}/${x}/${y}.png`;
}

function loadTile(x, y, z) {
    const key = `${z}/${x}/${y}`;

    if (state.tileCache.has(key)) {
        return state.tileCache.get(key);
    }

    if (state.loadingTiles.has(key)) {
        return null;
    }

    state.loadingTiles.add(key);

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
        state.tileCache.set(key, img);
        state.loadingTiles.delete(key);
        render();
    };
    img.onerror = () => {
        state.loadingTiles.delete(key);
    };
    img.src = getTileUrl(x, y, z);

    return null;
}

function drawMapTiles() {
    const ctx = state.ctx;
    const canvas = state.canvas;
    const zoom = Math.floor(state.zoomLevel);

    // Calculate scale factor for fractional zoom
    // When zoomLevel=13.5, tiles from zoom 13 should be drawn 1.41x larger (2^0.5)
    const zoomFraction = state.zoomLevel - zoom;
    const scaleFactor = Math.pow(2, zoomFraction);
    const scaledTileSize = state.tileSize * scaleFactor;

    // Calculate center tile using the integer zoom level
    const centerPixel = latLonToPixel(state.mapCenter.lat, state.mapCenter.lon, zoom);
    const centerTileX = Math.floor(centerPixel.x / state.tileSize);
    const centerTileY = Math.floor(centerPixel.y / state.tileSize);

    // Calculate offset within center tile, scaled for fractional zoom
    const offsetX = (centerPixel.x % state.tileSize) * scaleFactor;
    const offsetY = (centerPixel.y % state.tileSize) * scaleFactor;

    // Calculate how many tiles we need (account for larger scaled tiles)
    const tilesX = Math.ceil(canvas.width / scaledTileSize) + 2;
    const tilesY = Math.ceil(canvas.height / scaledTileSize) + 2;

    // Draw tiles
    for (let dx = -Math.floor(tilesX / 2); dx <= Math.ceil(tilesX / 2); dx++) {
        for (let dy = -Math.floor(tilesY / 2); dy <= Math.ceil(tilesY / 2); dy++) {
            const tileX = centerTileX + dx;
            const tileY = centerTileY + dy;

            // Skip invalid tiles
            const maxTile = Math.pow(2, zoom);
            if (tileX < 0 || tileX >= maxTile || tileY < 0 || tileY >= maxTile) {
                continue;
            }

            const screenX = canvas.width / 2 - offsetX + dx * scaledTileSize + state.viewOffset.x;
            const screenY = canvas.height / 2 - offsetY + dy * scaledTileSize + state.viewOffset.y;

            const tile = loadTile(tileX, tileY, zoom);

            if (tile) {
                ctx.drawImage(tile, screenX, screenY, scaledTileSize, scaledTileSize);
            } else {
                // Draw placeholder
                ctx.fillStyle = '#2a3a4a';
                ctx.fillRect(screenX, screenY, scaledTileSize, scaledTileSize);
                ctx.strokeStyle = '#3a4a5a';
                ctx.strokeRect(screenX, screenY, scaledTileSize, scaledTileSize);
            }
        }
    }

    // Draw semi-transparent overlay for better node visibility
    ctx.fillStyle = 'rgba(26, 26, 46, 0.3)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// ============== API Calls ==============
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        state.config = await response.json();

        document.getElementById('pathloss-model').value = state.config.model;
        document.getElementById('area-width').value = state.config.xsize;
        document.getElementById('area-height').value = state.config.ysize;
    } catch (error) {
        log('Failed to load config', 'error');
    }
}

async function loadNodes() {
    try {
        const response = await fetch('/api/nodes');
        const nodes = await response.json();

        state.nodes.clear();
        nodes.forEach(node => {
            state.nodes.set(node.id, node);
        });

        // Fetch topology to get RF-calculated links
        await refreshTopology();

        updateNodeList();
        updateCommandSelects();
        refreshStatistics();
        render();

        // Preload tiles around current view
        setTimeout(preloadTilesAroundView, 500);
    } catch (error) {
        log('Failed to load nodes', 'error');
    }
}

async function refreshTopology() {
    try {
        const topology = await getTopology();
        state.links = topology.links || [];
        debugLog(`Loaded ${state.links.length} RF links`);
    } catch (error) {
        console.error('Failed to refresh topology:', error);
        state.links = [];
    }
}

async function addNode(x, y) {
    const name = document.getElementById('node-name').value || `Node ${state.nodes.size}`;
    const height = parseFloat(document.getElementById('node-height').value) || 1;
    const gain = parseFloat(document.getElementById('node-gain').value) || 0;
    const role = document.getElementById('node-role').value;
    const hopLimit = parseInt(document.getElementById('node-hoplimit').value) || 3;

    try {
        const response = await fetch('/api/nodes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x, y, z: height, role, hopLimit, antennaGain: gain, name })
        });

        if (!response.ok) throw new Error('Failed to add node');

        const node = await response.json();
        state.nodes.set(node.id, node);

        // Refresh RF links since topology changed
        await refreshTopology();

        updateNodeList();
        updateCommandSelects();
        render();

        // Clear name input for next node
        document.getElementById('node-name').value = '';
    } catch (error) {
        log('Failed to add node: ' + error.message, 'error');
    }
}

function addNodeFromForm() {
    const useGPS = document.getElementById('use-gps-coords').checked;
    let x, y;

    if (useGPS) {
        const lat = parseFloat(document.getElementById('node-lat').value);
        const lon = parseFloat(document.getElementById('node-lon').value);

        if (isNaN(lat) || isNaN(lon) || lat === 0 || lon === 0) {
            log('Please enter valid GPS coordinates', 'error');
            return;
        }

        // Convert to local meters relative to map center
        const coords = latLonToMeters(lat, lon, state.mapCenter.lat, state.mapCenter.lon);
        x = coords.x;
        y = coords.y;
    } else {
        x = parseFloat(document.getElementById('node-x').value) || 0;
        y = parseFloat(document.getElementById('node-y').value) || 0;
    }

    addNode(x, y);
}

async function updateNode(nodeId, updates) {
    try {
        const response = await fetch(`/api/nodes/${nodeId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates)
        });

        if (!response.ok) throw new Error('Failed to update node');

        const node = await response.json();
        state.nodes.set(node.id, node);

        // Refresh RF links since topology may have changed (position, height, gain)
        await refreshTopology();

        updateNodeList();
        render();
        return node;
    } catch (error) {
        log('Failed to update node: ' + error.message, 'error');
        return null;
    }
}

async function deleteNode(nodeId) {
    try {
        await fetch(`/api/nodes/${nodeId}`, { method: 'DELETE' });
        state.nodes.delete(nodeId);

        // Refresh RF links since topology changed
        await refreshTopology();

        updateNodeList();
        render();
    } catch (error) {
        log('Failed to delete node: ' + error.message, 'error');
    }
}

async function getTopology() {
    try {
        const response = await fetch('/api/topology');
        return await response.json();
    } catch (error) {
        log('Failed to get topology', 'error');
        return { nodes: [], links: [] };
    }
}

async function getLinkQuality(node1Id, node2Id) {
    try {
        const response = await fetch(`/api/link/${node1Id}/${node2Id}`);
        return await response.json();
    } catch (error) {
        return null;
    }
}

// ============== Canvas Rendering ==============
let renderCount = 0;
function render() {
    if (!state.ctx || !state.canvas) {
        console.error('Canvas not initialized');
        return;
    }

    const ctx = state.ctx;
    const canvas = state.canvas;

    if (renderCount < 3) {
        debugLog(`Render #${renderCount + 1}: canvas=${canvas.width}x${canvas.height}, nodes=${state.nodes.size}`);
        renderCount++;
    }

    // Clear canvas
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw map tiles
    drawMapTiles();

    // Draw grid overlay
    drawGrid();

    // Draw links
    drawLinks();

    // Draw highlighted route
    if (state.highlightedRoute && state.highlightedRoute.length > 1) {
        drawRoute(state.highlightedRoute);
    }

    // Draw nodes
    state.nodes.forEach(node => {
        drawNode(node);
    });

    // Draw drag indicator if dragging a node
    if (state.dragMode === 'node' && state.draggedNode) {
        drawDragIndicator();
    }

    // Draw animations (packets, indicators)
    drawAnimations();

    // Draw scale bar
    drawScaleBar();
}

function drawGrid() {
    const ctx = state.ctx;
    const canvas = state.canvas;
    const mpp = metersPerPixel(state.mapCenter.lat, state.zoomLevel);

    // Calculate appropriate grid spacing based on zoom
    let gridSpacing;
    if (mpp > 100) gridSpacing = 10000;      // 10km
    else if (mpp > 50) gridSpacing = 5000;   // 5km
    else if (mpp > 20) gridSpacing = 2000;   // 2km
    else if (mpp > 10) gridSpacing = 1000;   // 1km
    else if (mpp > 5) gridSpacing = 500;     // 500m
    else if (mpp > 2) gridSpacing = 200;     // 200m
    else gridSpacing = 100;                   // 100m

    const screenSpacing = gridSpacing / mpp;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;

    // Calculate grid offset
    const worldCenter = { x: 0, y: 0 };
    const screenCenter = worldToScreen(worldCenter.x, worldCenter.y);

    const offsetX = ((screenCenter.x % screenSpacing) + screenSpacing) % screenSpacing;
    const offsetY = ((screenCenter.y % screenSpacing) + screenSpacing) % screenSpacing;

    // Vertical lines
    for (let x = offsetX; x < canvas.width; x += screenSpacing) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }

    // Horizontal lines
    for (let y = offsetY; y < canvas.height; y += screenSpacing) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
}

function drawScaleBar() {
    const ctx = state.ctx;
    const mpp = metersPerPixel(state.mapCenter.lat, state.zoomLevel);

    // Choose appropriate scale based on zoom
    let scaleMeters;
    if (mpp > 100) scaleMeters = 10000;      // 10km
    else if (mpp > 50) scaleMeters = 5000;   // 5km
    else if (mpp > 20) scaleMeters = 2000;   // 2km
    else if (mpp > 10) scaleMeters = 1000;   // 1km
    else if (mpp > 5) scaleMeters = 500;     // 500m
    else if (mpp > 2) scaleMeters = 200;     // 200m
    else scaleMeters = 100;                   // 100m

    const barLength = scaleMeters / mpp;

    const x = 20;
    const y = state.canvas.height - 30;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(x - 5, y - 20, barLength + 30, 35);

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + barLength, y);
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x, y + 5);
    ctx.moveTo(x + barLength, y - 5);
    ctx.lineTo(x + barLength, y + 5);
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    const label = scaleMeters >= 1000 ? `${scaleMeters/1000}km` : `${scaleMeters}m`;
    ctx.fillText(label, x + barLength / 2, y - 8);
}

function drawLinks() {
    const ctx = state.ctx;

    // Draw links from server-calculated RF propagation data
    state.links.forEach(link => {
        const n1 = state.nodes.get(link.source);
        const n2 = state.nodes.get(link.target);

        if (!n1 || !n2) return;

        const p1 = worldToScreen(n1.x, n1.y);
        const p2 = worldToScreen(n2.x, n2.y);

        // Use signal quality from server (0-100)
        const quality = link.signalQuality / 100;

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);

        // Color based on signal quality
        if (quality > 0.5) {
            // Good signal (green)
            ctx.strokeStyle = `rgba(15, 155, 15, ${0.4 + quality * 0.4})`;
        } else if (quality > 0.2) {
            // Medium signal (yellow/orange)
            ctx.strokeStyle = `rgba(243, 156, 18, ${0.4 + quality * 0.4})`;
        } else {
            // Weak signal (red) - but still receivable
            ctx.strokeStyle = `rgba(233, 69, 96, ${0.3 + quality * 0.3})`;
        }

        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

function drawRoute(route) {
    const ctx = state.ctx;

    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 4;
    ctx.setLineDash([10, 5]);

    ctx.beginPath();
    for (let i = 0; i < route.length; i++) {
        const node = state.nodes.get(route[i]);
        if (node) {
            const pos = worldToScreen(node.x, node.y);
            if (i === 0) {
                ctx.moveTo(pos.x, pos.y);
            } else {
                ctx.lineTo(pos.x, pos.y);
            }
        }
    }
    ctx.stroke();
    ctx.setLineDash([]);
}

function drawNode(node) {
    const ctx = state.ctx;
    const pos = worldToScreen(node.x, node.y);

    const isSelected = state.selectedNodes.has(node.id);
    const isDragging = state.draggedNode && state.draggedNode.id === node.id;
    const radius = isSelected ? 14 : 12;

    // Coverage area
    const mpp = metersPerPixel(state.mapCenter.lat, state.zoomLevel);
    if (node.coverageRadius && node.coverageRadius / mpp > 20) {
        const coverageScreenRadius = node.coverageRadius / mpp;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, coverageScreenRadius, 0, Math.PI * 2);
        ctx.fillStyle = getRoleColor(node.role, 0.15);
        ctx.fill();
        ctx.strokeStyle = getRoleColor(node.role, 0.3);
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Node shadow
    ctx.beginPath();
    ctx.arc(pos.x + 2, pos.y + 2, radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fill();

    // Node circle
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

    // Fill based on role
    ctx.fillStyle = getRoleColor(node.role);
    ctx.fill();

    // Border
    if (isDragging) {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 4;
    } else if (isSelected) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3;
    } else {
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.lineWidth = 2;
    }
    ctx.stroke();

    // Label background
    const labelText = node.name || `Node ${node.id}`;
    ctx.font = 'bold 12px sans-serif';
    const textWidth = ctx.measureText(labelText).width;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(pos.x - textWidth/2 - 4, pos.y - radius - 22, textWidth + 8, 18);

    // Label text
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.fillText(labelText, pos.x, pos.y - radius - 8);

    // Role indicator for routers/repeaters
    if (node.role === 'ROUTER' || node.role === 'REPEATER') {
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px sans-serif';
        ctx.fillText(node.role === 'ROUTER' ? 'R' : 'P', pos.x, pos.y + 4);
    }
}

function drawDragIndicator() {
    const ctx = state.ctx;
    const node = state.draggedNode;
    const pos = worldToScreen(node.x, node.y);

    // Draw crosshair
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    ctx.beginPath();
    ctx.moveTo(pos.x - 30, pos.y);
    ctx.lineTo(pos.x + 30, pos.y);
    ctx.moveTo(pos.x, pos.y - 30);
    ctx.lineTo(pos.x, pos.y + 30);
    ctx.stroke();

    ctx.setLineDash([]);

    // Draw coordinates
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(pos.x + 20, pos.y + 20, 120, 40);
    ctx.fillStyle = '#00ff00';
    ctx.font = '11px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`X: ${Math.round(node.x)}m`, pos.x + 25, pos.y + 35);
    ctx.fillText(`Y: ${Math.round(node.y)}m`, pos.x + 25, pos.y + 50);
}

function getRoleColor(role, alpha = 1) {
    const colors = {
        'CLIENT': `rgba(52, 152, 219, ${alpha})`,
        'CLIENT_MUTE': `rgba(149, 165, 166, ${alpha})`,
        'ROUTER': `rgba(243, 156, 18, ${alpha})`,
        'REPEATER': `rgba(46, 204, 113, ${alpha})`
    };
    return colors[role] || colors['CLIENT'];
}

// ============== Canvas Event Handlers ==============
function handleCanvasClick(e) {
    // If animation is active, ensure render loop is running
    if (state.animation.active) {
        startRenderLoop();
    }

    // Don't process click if we just finished dragging a node
    if (state.dragMode === 'node') {
        return;
    }

    const rect = state.canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    // Check if clicked on a node
    const clickedNode = findNodeAtPosition(screenX, screenY);

    if (clickedNode) {
        if (e.shiftKey) {
            // Multi-select with shift
            if (state.selectedNodes.has(clickedNode.id)) {
                state.selectedNodes.delete(clickedNode.id);
            } else {
                state.selectedNodes.add(clickedNode.id);
            }
        } else {
            state.selectedNodes.clear();
            state.selectedNodes.add(clickedNode.id);
        }

        updateLinkInfo();
        updateNodeList();
        render();
    } else if (!e.shiftKey) {
        state.selectedNodes.clear();
        updateLinkInfo();
        updateNodeList();
        render();
    }
}

function handleCanvasDoubleClick(e) {
    const rect = state.canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    const clickedNode = findNodeAtPosition(screenX, screenY);

    if (clickedNode) {
        // Open edit modal
        openEditModal(clickedNode);
    } else {
        // Add new node at clicked position
        const worldPos = screenToWorld(screenX, screenY);
        addNode(Math.round(worldPos.x), Math.round(worldPos.y));
    }
}

function handleMouseDown(e) {
    if (e.button !== 0) return; // Left click only

    const rect = state.canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    // Check if clicking on a node
    const clickedNode = findNodeAtPosition(screenX, screenY);

    if (clickedNode) {
        // Start dragging the node
        state.dragMode = 'node';
        state.draggedNode = { ...clickedNode }; // Clone node data
        state.isDragging = true;
        state.canvas.style.cursor = 'grabbing';
    } else {
        // Start panning
        state.dragMode = 'pan';
        state.isDragging = true;
        state.canvas.style.cursor = 'grabbing';
    }

    state.lastMouse = { x: e.clientX, y: e.clientY };
}

function handleMouseMove(e) {
    const rect = state.canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    // Update cursor position display
    const worldPos = screenToWorld(screenX, screenY);
    const latLon = metersToLatLon(worldPos.x, worldPos.y, state.mapCenter.lat, state.mapCenter.lon);
    document.getElementById('cursor-pos').textContent =
        `${Math.round(worldPos.x)}m, ${Math.round(worldPos.y)}m | ${latLon.lat.toFixed(5)}°, ${latLon.lon.toFixed(5)}°`;

    // Update cursor style based on what's under it
    if (!state.isDragging) {
        const hoveredNode = findNodeAtPosition(screenX, screenY);
        state.canvas.style.cursor = hoveredNode ? 'grab' : 'crosshair';
    }

    if (state.isDragging) {
        const dx = e.clientX - state.lastMouse.x;
        const dy = e.clientY - state.lastMouse.y;

        if (state.dragMode === 'node' && state.draggedNode) {
            // Move the node
            const mpp = metersPerPixel(state.mapCenter.lat, state.zoomLevel);
            state.draggedNode.x += dx * mpp;
            state.draggedNode.y -= dy * mpp;

            // Update the node in state temporarily for rendering
            const node = state.nodes.get(state.draggedNode.id);
            if (node) {
                node.x = state.draggedNode.x;
                node.y = state.draggedNode.y;
            }
        } else {
            // Pan the map
            state.viewOffset.x += dx;
            state.viewOffset.y += dy;
        }

        state.lastMouse = { x: e.clientX, y: e.clientY };
        render();
    }
}

function handleMouseUp(e) {
    if (state.dragMode === 'node' && state.draggedNode && state.isDragging) {
        // Save the node position to server
        updateNode(state.draggedNode.id, {
            x: Math.round(state.draggedNode.x),
            y: Math.round(state.draggedNode.y)
        }).then(() => {
            log(`Moved ${state.draggedNode.name || 'Node ' + state.draggedNode.id} to (${Math.round(state.draggedNode.x)}, ${Math.round(state.draggedNode.y)})`, 'success');
        });

        state.draggedNode = null;

        // Prevent click event from firing
        setTimeout(() => {
            state.dragMode = 'pan';
        }, 100);
    }

    state.isDragging = false;
    state.canvas.style.cursor = 'crosshair';
}

function handleMouseLeave() {
    if (state.isDragging && state.dragMode === 'node' && state.draggedNode) {
        // Cancel node drag on leave
        const nodeName = state.draggedNode.name || 'Node ' + state.draggedNode.id;
        loadNodes(); // Reload to reset position
        state.draggedNode = null;
        log(`Move cancelled for ${nodeName} (dragged outside canvas)`, 'warning');
    }
    state.isDragging = false;
    state.dragMode = 'pan';
    state.canvas.style.cursor = 'crosshair';
}

function handleWheel(e) {
    e.preventDefault();

    const rect = state.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Get world position before zoom
    const worldBefore = screenToWorld(mouseX, mouseY);

    // Zoom
    const zoomDelta = e.deltaY < 0 ? 1 : -1;
    const newZoom = Math.max(1, Math.min(19, state.zoomLevel + zoomDelta * 0.5));

    if (newZoom !== state.zoomLevel) {
        state.zoomLevel = newZoom;

        // Adjust offset to zoom towards cursor
        const worldAfter = screenToWorld(mouseX, mouseY);
        const mpp = metersPerPixel(state.mapCenter.lat, state.zoomLevel);
        state.viewOffset.x += (worldAfter.x - worldBefore.x) / mpp;
        state.viewOffset.y -= (worldAfter.y - worldBefore.y) / mpp;

        updateZoomDisplay();
        render();
    }
}

// Touch event handlers for mobile
function handleTouchStart(e) {
    e.preventDefault();
    if (e.touches.length === 1) {
        const touch = e.touches[0];
        handleMouseDown({ button: 0, clientX: touch.clientX, clientY: touch.clientY });
    }
}

function handleTouchMove(e) {
    e.preventDefault();
    if (e.touches.length === 1) {
        const touch = e.touches[0];
        handleMouseMove({ clientX: touch.clientX, clientY: touch.clientY });
    }
}

function handleTouchEnd(e) {
    handleMouseUp({});
}

function findNodeAtPosition(screenX, screenY) {
    for (const node of state.nodes.values()) {
        const pos = worldToScreen(node.x, node.y);
        const dist = Math.sqrt(Math.pow(screenX - pos.x, 2) + Math.pow(screenY - pos.y, 2));
        if (dist < 20) { // Slightly larger hit area for easier selection
            return node;
        }
    }
    return null;
}

// ============== UI Updates ==============
function updateNodeList() {
    const container = document.getElementById('node-list');
    document.getElementById('node-count').textContent = state.nodes.size;

    container.innerHTML = '';

    state.nodes.forEach(node => {
        const item = document.createElement('div');
        item.className = 'node-item' + (state.selectedNodes.has(node.id) ? ' selected' : '');

        // Build DOM safely to prevent XSS
        const nodeInfo = document.createElement('div');
        nodeInfo.className = 'node-info';

        const nodeName = document.createElement('span');
        nodeName.className = 'node-name';
        nodeName.textContent = node.name || 'Node ' + node.id;

        const nodeDetails = document.createElement('span');
        nodeDetails.className = 'node-details';
        nodeDetails.textContent = `X: ${Math.round(node.x)}, Y: ${Math.round(node.y)}, H: ${node.z}m`;

        nodeInfo.appendChild(nodeName);

        // Add source indicator for imported nodes
        if (node.source === 'imported') {
            const sourceSpan = document.createElement('span');
            sourceSpan.className = 'node-source imported';
            sourceSpan.textContent = 'imported';
            nodeInfo.appendChild(sourceSpan);
        }

        nodeInfo.appendChild(nodeDetails);

        const nodeRole = document.createElement('span');
        nodeRole.className = 'node-role ' + (node.role || 'CLIENT').toLowerCase();
        nodeRole.textContent = node.role || 'CLIENT';

        item.appendChild(nodeInfo);
        item.appendChild(nodeRole);

        item.addEventListener('click', () => {
            state.selectedNodes.clear();
            state.selectedNodes.add(node.id);
            centerOnNode(node);
            updateNodeList();
            updateLinkInfo();
            render();
        });

        item.addEventListener('dblclick', () => {
            openEditModal(node);
        });

        container.appendChild(item);
    });
}

function updateCommandSelects() {
    const selects = [
        'cmd-broadcast-from',
        'cmd-dm-from', 'cmd-dm-to',
        'cmd-trace-from', 'cmd-trace-to'
    ];

    selects.forEach(id => {
        const select = document.getElementById(id);
        const currentValue = select.value;
        select.innerHTML = '';

        state.nodes.forEach(node => {
            const option = document.createElement('option');
            option.value = node.id;
            option.textContent = node.name || `Node ${node.id}`;
            select.appendChild(option);
        });

        // Restore selection if still valid
        if (select.querySelector(`option[value="${currentValue}"]`)) {
            select.value = currentValue;
        }
    });
}

function updateZoomDisplay() {
    document.getElementById('zoom-level').textContent = `Z${Math.round(state.zoomLevel)}`;
}

async function updateLinkInfo() {
    const container = document.getElementById('link-info');

    if (state.selectedNodes.size === 2) {
        const nodeIds = Array.from(state.selectedNodes);
        const quality = await getLinkQuality(nodeIds[0], nodeIds[1]);

        if (quality) {
            const qualityClass = quality.signalQuality > 60 ? 'good' :
                                quality.signalQuality > 30 ? 'poor' : 'bad';

            container.innerHTML = `
                <div class="link-stat">
                    <span>Distance:</span>
                    <span class="value">${quality.distance}m</span>
                </div>
                <div class="link-stat">
                    <span>Path Loss:</span>
                    <span class="value">${quality.pathLoss} dB</span>
                </div>
                <div class="link-stat">
                    <span>RSSI:</span>
                    <span class="value ${qualityClass}">${quality.rssi} dBm</span>
                </div>
                <div class="link-stat">
                    <span>SNR:</span>
                    <span class="value">${quality.snr} dB</span>
                </div>
                <div class="link-stat">
                    <span>Can Receive:</span>
                    <span class="value ${quality.canReceive ? 'good' : 'bad'}">
                        ${quality.canReceive ? 'Yes' : 'No'}
                    </span>
                </div>
                <div class="link-stat">
                    <span>Signal Quality:</span>
                    <span class="value ${qualityClass}">${quality.signalQuality}%</span>
                </div>
            `;
        } else {
            container.innerHTML = '<p class="hint">Could not calculate link quality</p>';
        }
    } else {
        container.innerHTML = '<p class="hint">Select two nodes to see link quality</p>';
    }
}

// ============== Modal Functions ==============
function openEditModal(node) {
    state.editingNode = node;

    document.getElementById('edit-node-id').textContent = node.id;
    document.getElementById('edit-name').value = node.name || '';
    document.getElementById('edit-x').value = node.x;
    document.getElementById('edit-y').value = node.y;
    document.getElementById('edit-height').value = node.z;
    document.getElementById('edit-gain').value = node.antennaGain || 0;
    document.getElementById('edit-role').value = node.role;
    document.getElementById('edit-hoplimit').value = node.hopLimit || 3;
    document.getElementById('edit-hoplimit-value').textContent = node.hopLimit || 3;

    document.getElementById('edit-modal').classList.remove('hidden');
}

function closeEditModal() {
    state.editingNode = null;
    document.getElementById('edit-modal').classList.add('hidden');
}

async function saveEditedNode() {
    if (!state.editingNode) return;

    const updates = {
        name: document.getElementById('edit-name').value,
        x: parseFloat(document.getElementById('edit-x').value),
        y: parseFloat(document.getElementById('edit-y').value),
        z: parseFloat(document.getElementById('edit-height').value),
        antennaGain: parseFloat(document.getElementById('edit-gain').value),
        role: document.getElementById('edit-role').value,
        hopLimit: parseInt(document.getElementById('edit-hoplimit').value)
    };

    await updateNode(state.editingNode.id, updates);
    closeEditModal();
    loadNodes();
}

async function deleteEditedNode() {
    if (!state.editingNode) return;

    if (confirm(`Delete node ${state.editingNode.name || state.editingNode.id}?`)) {
        await deleteNode(state.editingNode.id);
        closeEditModal();
    }
}

// ============== Commands ==============
function sendBroadcast() {
    const fromNodeId = parseInt(document.getElementById('cmd-broadcast-from').value);
    const text = document.getElementById('cmd-broadcast-text').value || 'Test message';

    if (isNaN(fromNodeId)) {
        log('Please add nodes first and select a source node', 'error');
        return;
    }

    // Get hop limit from source node
    const sourceNode = getNodeById(fromNodeId);
    const hopLimit = sourceNode ? (sourceNode.hopLimit || 3) : 3;

    debugLog('Sending broadcast:', { from: fromNodeId, text, hopLimit });
    log(`Sending broadcast from Node ${fromNodeId} (hop limit: ${hopLimit})...`, 'info');

    state.socket.emit('send_command', {
        command: 'broadcast',
        args: { from: fromNodeId, text, hopLimit }
    });
}

function sendDM() {
    const fromNode = parseInt(document.getElementById('cmd-dm-from').value);
    const toNode = parseInt(document.getElementById('cmd-dm-to').value);
    const text = document.getElementById('cmd-dm-text').value || 'Test message';

    if (isNaN(fromNode) || isNaN(toNode)) {
        log('Please add at least 2 nodes and select source/destination', 'error');
        return;
    }

    debugLog('Sending DM:', { from: fromNode, to: toNode, text });
    log(`Sending DM from Node ${fromNode} to Node ${toNode}...`, 'info');

    state.socket.emit('send_command', {
        command: 'dm',
        args: { from: fromNode, to: toNode, text }
    });
}

function sendTraceroute() {
    const fromNode = parseInt(document.getElementById('cmd-trace-from').value);
    const toNode = parseInt(document.getElementById('cmd-trace-to').value);

    if (isNaN(fromNode) || isNaN(toNode)) {
        log('Please add at least 2 nodes and select source/destination', 'error');
        return;
    }

    debugLog('Running traceroute:', { from: fromNode, to: toNode });
    log(`Running traceroute from Node ${fromNode} to Node ${toNode}...`, 'info');

    state.socket.emit('send_command', {
        command: 'traceroute',
        args: { from: fromNode, to: toNode }
    });
}

// ============== Animation Functions ==============

// Helper to get node by ID (handles type mismatches)
function getNodeById(id) {
    // Try direct lookup first
    let node = state.nodes.get(id);
    if (node) return node;

    // Try as number
    node = state.nodes.get(Number(id));
    if (node) return node;

    // Try as string
    node = state.nodes.get(String(id));
    if (node) return node;

    // Search through all nodes
    for (const [key, n] of state.nodes) {
        if (n.id === id || n.id === Number(id) || String(n.id) === String(id)) {
            return n;
        }
    }

    return null;
}

function startBroadcastAnimation(simulation) {
    debugLog('Starting broadcast animation:', simulation);

    // Reset animation state
    state.animation = {
        active: true,
        type: 'broadcast',
        data: simulation,
        startTime: performance.now(),
        packets: [],
        nodeStates: new Map(),
        currentHop: 0
    };

    // Set initial node states
    simulation.nodes.forEach(n => {
        state.animation.nodeStates.set(n.id, {
            status: n.status,
            hop: n.hop,
            rssi: n.rssi || 0,
            pulsePhase: 0
        });
    });

    // Render immediately to show initial state, then start animation loop
    render();
    renderLoopRunning = true;
    startAnimationFallback();
    requestAnimationFrame(animateBroadcast);
}

function animateBroadcast(timestamp) {
    if (!state.animation.active || state.animation.type !== 'broadcast') return;

    const elapsed = timestamp - state.animation.startTime;
    const hopDuration = 1000; // 1 second per hop
    const currentHop = Math.floor(elapsed / hopDuration);
    const hopProgress = (elapsed % hopDuration) / hopDuration;

    const simulation = state.animation.data;
    const propagation = simulation.propagation || [];

    // Update packets for current hop
    state.animation.packets = [];

    if (currentHop < propagation.length) {
        const hopData = propagation[currentHop];
        hopData.transmissions.forEach(tx => {
            const fromNode = getNodeById(tx.from);
            const toNode = getNodeById(tx.to);
            if (fromNode && toNode) {
                state.animation.packets.push({
                    fromX: fromNode.x,
                    fromY: fromNode.y,
                    toX: toNode.x,
                    toY: toNode.y,
                    progress: hopProgress,
                    rssi: tx.rssi
                });
            } else {
                console.warn('Could not find nodes for broadcast hop:', tx.from, '->', tx.to);
            }
        });

        // Update node states - mark nodes as received when packet arrives
        if (hopProgress > 0.8) {
            hopData.transmissions.forEach(tx => {
                const nodeState = state.animation.nodeStates.get(tx.to);
                if (nodeState && nodeState.status !== 'received') {
                    nodeState.status = 'receiving';
                    nodeState.pulsePhase = 1;
                }
            });
        }
    }

    // Update pulse phases
    state.animation.nodeStates.forEach((nodeState, nodeId) => {
        if (nodeState.pulsePhase > 0) {
            nodeState.pulsePhase = Math.max(0, nodeState.pulsePhase - 0.02);
        }
    });

    state.animation.currentHop = currentHop;
    render();

    // Continue or end animation
    const totalDuration = (propagation.length + 1) * hopDuration + 2000;
    if (elapsed < totalDuration) {
        requestAnimationFrame(animateBroadcast);
    } else {
        // End animation - show final state for a moment
        setTimeout(() => {
            state.animation.active = false;
            state.animation.packets = [];
            renderLoopRunning = false;
            render();
        }, 1000);
    }
}

function startDMAnimation(simulation) {
    debugLog('Starting DM animation:', simulation);

    // DM now uses flood propagation like broadcast
    // Show the flood, then highlight the path to destination (if reached)

    // Get target node from destination field (works even when path is empty)
    const targetNode = simulation.destination !== undefined ? simulation.destination :
                       (simulation.path && simulation.path.length > 0 ? simulation.path[simulation.path.length - 1] : null);

    state.animation = {
        active: true,
        type: 'dm',
        data: simulation,
        startTime: performance.now(),
        packets: [],
        nodeStates: new Map(),
        currentHop: 0,
        destinationReached: simulation.delivered,
        targetNode: targetNode
    };

    // If we have flood/propagation data, use broadcast-style animation
    if (simulation.propagation && simulation.propagation.length > 0) {
        // Set up node states from flood result
        const floodResult = simulation.floodResult || simulation;
        if (floodResult.nodes) {
            floodResult.nodes.forEach(n => {
                state.animation.nodeStates.set(n.id, {
                    status: n.status,
                    hop: n.hop,
                    rssi: n.rssi || 0,
                    pulsePhase: 0,
                    isTarget: n.id === state.animation.targetNode
                });
            });
        }

        // Render immediately to show initial state, then start animation loop
        render();
        renderLoopRunning = true;
        startAnimationFallback();
        requestAnimationFrame(animateDMFlood);
    } else if (simulation.delivered && simulation.hops && simulation.hops.length > 0) {
        // Fallback to old path-based animation
        state.highlightedRoute = simulation.path || [];
        render();
        renderLoopRunning = true;
        startAnimationFallback();
        requestAnimationFrame(animateDM);
    } else {
        log(`DM failed - ${simulation.reason || 'no route available'}`, 'error');
        state.animation.active = false;
        renderLoopRunning = false;
    }
}

function animateDMFlood(timestamp) {
    // DM flood animation - shows message flooding out in all directions
    if (!state.animation.active || state.animation.type !== 'dm') return;

    const elapsed = timestamp - state.animation.startTime;
    const hopDuration = 1000;
    const currentHop = Math.floor(elapsed / hopDuration);
    const hopProgress = (elapsed % hopDuration) / hopDuration;

    const simulation = state.animation.data;
    const propagation = simulation.propagation || [];

    // Update packets for current hop
    state.animation.packets = [];

    if (currentHop < propagation.length) {
        const hopData = propagation[currentHop];
        hopData.transmissions.forEach(tx => {
            const fromNode = getNodeById(tx.from);
            const toNode = getNodeById(tx.to);
            if (fromNode && toNode) {
                state.animation.packets.push({
                    fromX: fromNode.x,
                    fromY: fromNode.y,
                    toX: toNode.x,
                    toY: toNode.y,
                    progress: hopProgress,
                    rssi: tx.rssi,
                    isDM: true
                });
            }
        });

        // Update node states
        if (hopProgress > 0.8) {
            hopData.transmissions.forEach(tx => {
                const nodeState = state.animation.nodeStates.get(tx.to);
                if (nodeState && nodeState.status !== 'received') {
                    nodeState.status = 'receiving';
                    nodeState.pulsePhase = 1;
                }
            });
        }
    }

    // Update pulse phases
    state.animation.nodeStates.forEach((nodeState) => {
        if (nodeState.pulsePhase > 0) {
            nodeState.pulsePhase = Math.max(0, nodeState.pulsePhase - 0.02);
        }
    });

    state.animation.currentHop = currentHop;
    render();

    // Continue or end animation
    const totalDuration = (propagation.length + 1) * hopDuration + 2000;
    if (elapsed < totalDuration) {
        requestAnimationFrame(animateDMFlood);
    } else {
        // Show final result
        if (state.animation.destinationReached) {
            state.highlightedRoute = simulation.path;
            log(`DM delivered to destination via ${simulation.path.length - 1} hops`, 'success');
        } else {
            log(`DM FAILED - message died after ${simulation.hopLimit} hops, destination not reached`, 'error');
        }

        setTimeout(() => {
            state.animation.active = false;
            state.animation.packets = [];
            state.highlightedRoute = null;
            renderLoopRunning = false;
            render();
        }, 2000);
    }
}

function animateDM(timestamp) {
    // Legacy path-based DM animation (fallback)
    if (!state.animation.active || state.animation.type !== 'dm') return;

    const elapsed = timestamp - state.animation.startTime;
    const simulation = state.animation.data;
    const hops = simulation.hops || [];
    const hopDuration = 800;

    state.animation.packets = [];

    if (simulation.delivered && hops.length > 0) {
        const totalProgress = elapsed / hopDuration;
        const currentHopIndex = Math.floor(totalProgress);
        const hopProgress = totalProgress - currentHopIndex;

        if (currentHopIndex < hops.length) {
            const hop = hops[currentHopIndex];
            const fromNode = getNodeById(hop.from);
            const toNode = getNodeById(hop.to);

            if (fromNode && toNode) {
                state.animation.packets.push({
                    fromX: fromNode.x,
                    fromY: fromNode.y,
                    toX: toNode.x,
                    toY: toNode.y,
                    progress: Math.min(1, hopProgress),
                    rssi: hop.rssi,
                    isDM: true
                });
            } else {
                console.warn('Could not find nodes for DM hop:', hop.from, '->', hop.to, 'Available nodes:', Array.from(state.nodes.keys()));
            }
        }
    }

    render();

    const totalDuration = hops.length * hopDuration + 1500;
    if (elapsed < totalDuration) {
        requestAnimationFrame(animateDM);
    } else {
        state.animation.active = false;
        state.animation.packets = [];
        renderLoopRunning = false;
        // Keep route highlighted a bit longer
        setTimeout(() => {
            state.highlightedRoute = null;
            render();
        }, 2000);
    }
}

function startTracerouteAnimation(simulation) {
    debugLog('Starting traceroute animation:', simulation);

    // Get target node from destination field (works even when path is empty)
    const targetNode = simulation.destination !== undefined ? simulation.destination :
                       (simulation.path && simulation.path.length > 0 ? simulation.path[simulation.path.length - 1] : null);

    state.animation = {
        active: true,
        type: 'traceroute',
        data: simulation,
        startTime: performance.now(),
        packets: [],
        nodeStates: new Map(),
        currentHop: 0,
        discoveredHops: [],
        destinationReached: simulation.reachable,
        targetNode: targetNode
    };

    state.highlightedRoute = [];

    // If we have flood/propagation data, use flood-style animation
    if (simulation.propagation && simulation.propagation.length > 0) {
        const floodResult = simulation.floodResult || simulation;
        if (floodResult.nodes) {
            floodResult.nodes.forEach(n => {
                state.animation.nodeStates.set(n.id, {
                    status: n.status,
                    hop: n.hop,
                    rssi: n.rssi || 0,
                    pulsePhase: 0,
                    isTarget: n.id === targetNode
                });
            });
        }
        // Render immediately to show initial state, then start animation loop
        render();
        renderLoopRunning = true;
        startAnimationFallback();
        requestAnimationFrame(animateTracerouteFlood);
    } else {
        render();
        renderLoopRunning = true;
        startAnimationFallback();
        requestAnimationFrame(animateTraceroute);
    }
}

function animateTracerouteFlood(timestamp) {
    // Traceroute flood animation - shows message flooding out
    if (!state.animation.active || state.animation.type !== 'traceroute') return;

    const elapsed = timestamp - state.animation.startTime;
    const hopDuration = 800;
    const currentHop = Math.floor(elapsed / hopDuration);
    const hopProgress = (elapsed % hopDuration) / hopDuration;

    const simulation = state.animation.data;
    const propagation = simulation.propagation || [];

    state.animation.packets = [];

    if (currentHop < propagation.length) {
        const hopData = propagation[currentHop];
        hopData.transmissions.forEach(tx => {
            const fromNode = getNodeById(tx.from);
            const toNode = getNodeById(tx.to);
            if (fromNode && toNode) {
                state.animation.packets.push({
                    fromX: fromNode.x,
                    fromY: fromNode.y,
                    toX: toNode.x,
                    toY: toNode.y,
                    progress: hopProgress,
                    rssi: tx.rssi,
                    isTraceroute: true
                });
            }
        });

        if (hopProgress > 0.8) {
            hopData.transmissions.forEach(tx => {
                const nodeState = state.animation.nodeStates.get(tx.to);
                if (nodeState && nodeState.status !== 'received') {
                    nodeState.status = 'receiving';
                    nodeState.pulsePhase = 1;
                }
            });
        }
    }

    state.animation.nodeStates.forEach((nodeState) => {
        if (nodeState.pulsePhase > 0) {
            nodeState.pulsePhase = Math.max(0, nodeState.pulsePhase - 0.02);
        }
    });

    state.animation.currentHop = currentHop;
    render();

    const totalDuration = (propagation.length + 1) * hopDuration + 2000;
    if (elapsed < totalDuration) {
        requestAnimationFrame(animateTracerouteFlood);
    } else {
        if (state.animation.destinationReached) {
            state.highlightedRoute = simulation.path;
            log(`Traceroute completed - destination reached in ${simulation.path.length - 1} hops`, 'success');
        } else {
            log(`Traceroute FAILED - message died after ${simulation.hopLimit} hops, destination not reached`, 'error');
        }

        setTimeout(() => {
            state.animation.active = false;
            state.animation.packets = [];
            state.highlightedRoute = null;
            renderLoopRunning = false;
            render();
        }, 3000);
    }
}

function animateTraceroute(timestamp) {
    // Legacy traceroute animation (fallback)
    if (!state.animation.active || state.animation.type !== 'traceroute') return;

    const elapsed = timestamp - state.animation.startTime;
    const simulation = state.animation.data;
    const hops = simulation.hops || [];
    const hopDuration = 600;

    state.animation.packets = [];

    if (simulation.reachable && hops.length > 1) {
        const totalProgress = elapsed / hopDuration;
        const currentHopIndex = Math.floor(totalProgress);
        const hopProgress = totalProgress - currentHopIndex;

        // Update discovered path
        const discoveredPath = hops.slice(0, currentHopIndex + 1).map(h => h.node);
        state.highlightedRoute = discoveredPath;

        // Show packet traveling
        if (currentHopIndex < hops.length - 1) {
            const fromHop = hops[currentHopIndex];
            const toHop = hops[currentHopIndex + 1];
            const fromNode = getNodeById(fromHop.node);
            const toNode = getNodeById(toHop.node);

            if (fromNode && toNode) {
                state.animation.packets.push({
                    fromX: fromNode.x,
                    fromY: fromNode.y,
                    toX: toNode.x,
                    toY: toNode.y,
                    progress: Math.min(1, hopProgress),
                    rssi: toHop.rssi,
                    isTraceroute: true
                });
            } else {
                console.warn('Could not find nodes for traceroute hop:', fromHop.node, '->', toHop.node);
            }

            // Log hop discovery
            if (hopProgress < 0.1 && currentHopIndex > state.animation.currentHop) {
                const hopInfo = toHop;
                log(`  Hop ${currentHopIndex + 1}: Node ${hopInfo.node} (${hopInfo.name}) - ${hopInfo.rssi.toFixed(0)}dBm, ${hopInfo.distance.toFixed(0)}m`, 'info');
            }
        }

        state.animation.currentHop = currentHopIndex;
    }

    render();

    const totalDuration = hops.length * hopDuration + 2000;
    if (elapsed < totalDuration) {
        requestAnimationFrame(animateTraceroute);
    } else {
        state.animation.active = false;
        state.animation.packets = [];
        renderLoopRunning = false;
        // Keep final route visible
        state.highlightedRoute = simulation.path;
        render();
        setTimeout(() => {
            state.highlightedRoute = null;
            render();
        }, 3000);
    }
}

function drawAnimations() {
    if (!state.animation.active) return;

    const ctx = state.ctx;

    // Draw packets
    state.animation.packets.forEach(packet => {
        const fromPos = worldToScreen(packet.fromX, packet.fromY);
        const toPos = worldToScreen(packet.toX, packet.toY);

        // Calculate current position
        const x = fromPos.x + (toPos.x - fromPos.x) * packet.progress;
        const y = fromPos.y + (toPos.y - fromPos.y) * packet.progress;

        // Draw packet trail
        const gradient = ctx.createLinearGradient(fromPos.x, fromPos.y, x, y);
        gradient.addColorStop(0, 'rgba(0, 255, 255, 0)');
        gradient.addColorStop(1, 'rgba(0, 255, 255, 0.8)');

        ctx.strokeStyle = gradient;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(fromPos.x, fromPos.y);
        ctx.lineTo(x, y);
        ctx.stroke();

        // Draw packet as glowing circle
        const packetColor = packet.isDM ? '#00ff00' : packet.isTraceroute ? '#ffff00' : '#00ffff';

        // Convert hex color to rgba with alpha
        function hexToRgba(hex, alpha) {
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }

        // Outer glow
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fillStyle = hexToRgba(packetColor, 0.3);
        ctx.shadowColor = packetColor;
        ctx.shadowBlur = 15;
        ctx.fill();

        // Inner packet
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = packetColor;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Show RSSI near packet
        if (packet.rssi) {
            ctx.font = '10px monospace';
            ctx.fillStyle = '#ffffff';
            ctx.textAlign = 'left';
            ctx.fillText(`${packet.rssi.toFixed(0)}dBm`, x + 10, y - 5);
        }
    });

    // Draw node status indicators for broadcast/DM/traceroute animations
    if (state.animation.type === 'broadcast' || state.animation.type === 'dm' || state.animation.type === 'traceroute') {
        state.animation.nodeStates.forEach((nodeState, nodeId) => {
            const node = getNodeById(nodeId);
            if (!node) return;

            const pos = worldToScreen(node.x, node.y);

            if (nodeState.status === 'source') {
                // Pulsing source indicator
                const pulse = Math.sin(performance.now() / 200) * 0.3 + 0.7;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 25 * pulse, 0, Math.PI * 2);
                ctx.strokeStyle = `rgba(255, 200, 0, ${pulse})`;
                ctx.lineWidth = 3;
                ctx.stroke();
            } else if (nodeState.status === 'received' || nodeState.status === 'receiving') {
                // Received indicator - green ring (or highlight if it's the target)
                const alpha = nodeState.pulsePhase > 0 ? nodeState.pulsePhase : 0.5;
                const isTarget = nodeState.isTarget;

                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);
                if (isTarget) {
                    ctx.strokeStyle = `rgba(0, 255, 255, ${alpha + 0.3})`;  // Cyan for target
                    ctx.lineWidth = 4;
                } else {
                    ctx.strokeStyle = `rgba(0, 255, 100, ${alpha})`;
                    ctx.lineWidth = 2;
                }
                ctx.stroke();

                // Show hop count
                ctx.font = 'bold 10px sans-serif';
                ctx.fillStyle = isTarget ? '#00ffff' : '#00ff64';
                ctx.textAlign = 'center';
                ctx.fillText(`H${nodeState.hop}`, pos.x, pos.y - 22);

                // Mark target with a star/indicator
                if (isTarget) {
                    ctx.fillStyle = '#00ffff';
                    ctx.fillText('★ TARGET', pos.x, pos.y + 30);
                }
            } else if (nodeState.status === 'unreached') {
                // Unreached indicator - red X (only show after animation mostly done)
                const propagationLength = state.animation.data.propagation?.length || 0;
                if (state.animation.currentHop >= propagationLength) {
                    ctx.strokeStyle = 'rgba(255, 50, 50, 0.7)';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(pos.x - 8, pos.y - 8);
                    ctx.lineTo(pos.x + 8, pos.y + 8);
                    ctx.moveTo(pos.x + 8, pos.y - 8);
                    ctx.lineTo(pos.x - 8, pos.y + 8);
                    ctx.stroke();

                    // If this is the target that wasn't reached, show it clearly
                    if (nodeState.isTarget) {
                        ctx.fillStyle = '#ff3333';
                        ctx.font = 'bold 10px sans-serif';
                        ctx.textAlign = 'center';
                        ctx.fillText('✗ TARGET NOT REACHED', pos.x, pos.y + 30);
                    }
                }
            }
        });
    }
}

// ============== Import/Export ==============
async function exportYaml() {
    try {
        const response = await fetch('/api/export/yaml');
        const yaml = await response.text();

        // Create download
        const blob = new Blob([yaml], { type: 'text/yaml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'nodeConfig.yaml';
        a.click();
        URL.revokeObjectURL(url);

        log('Configuration exported', 'success');
    } catch (error) {
        log('Export failed: ' + error.message, 'error');
    }
}

async function importYaml() {
    const yaml = document.getElementById('import-yaml').value;

    if (!yaml.trim()) {
        log('Please paste YAML configuration', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/import/yaml', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ yaml })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Import failed');
        }

        const result = await response.json();
        log(`Imported ${result.nodeCount} nodes`, 'success');

        document.getElementById('import-modal').classList.add('hidden');
        document.getElementById('import-yaml').value = '';

        loadNodes();
        fitAllNodes();
    } catch (error) {
        log('Import failed: ' + error.message, 'error');
    }
}

// ============== Utility Functions ==============
function fitAllNodes() {
    if (state.nodes.size === 0) {
        state.viewOffset = { x: 0, y: 0 };
        state.zoomLevel = 12;
        updateZoomDisplay();
        render();
        return;
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    state.nodes.forEach(node => {
        minX = Math.min(minX, node.x);
        maxX = Math.max(maxX, node.x);
        minY = Math.min(minY, node.y);
        maxY = Math.max(maxY, node.y);
    });

    const width = maxX - minX || 1000;
    const height = maxY - minY || 1000;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    // Calculate zoom level to fit all nodes
    const padding = 1.5;
    for (let z = 19; z >= 1; z--) {
        const mpp = metersPerPixel(state.mapCenter.lat, z);
        const screenWidth = width * padding / mpp;
        const screenHeight = height * padding / mpp;

        if (screenWidth < state.canvas.width && screenHeight < state.canvas.height) {
            state.zoomLevel = z;
            break;
        }
    }

    // Center on nodes
    const mpp = metersPerPixel(state.mapCenter.lat, state.zoomLevel);
    state.viewOffset.x = -centerX / mpp;
    state.viewOffset.y = centerY / mpp;

    updateZoomDisplay();
    render();
}

function centerOnNode(node) {
    const mpp = metersPerPixel(state.mapCenter.lat, state.zoomLevel);
    state.viewOffset.x = -node.x / mpp;
    state.viewOffset.y = node.y / mpp;
    render();
}

function applySettings() {
    const model = parseInt(document.getElementById('pathloss-model').value);
    const width = parseInt(document.getElementById('area-width').value);
    const height = parseInt(document.getElementById('area-height').value);

    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, xsize: width, ysize: height })
    }).then(() => {
        log('Settings applied', 'success');
        loadNodes(); // Recalculate coverage
    }).catch(() => {
        log('Failed to apply settings', 'error');
    });
}

// HTML escape function to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============== NodeDB Import Functions ==============

function getNodeDBFilters() {
    const roles = [];
    document.querySelectorAll('.filter-role:checked').forEach(cb => {
        roles.push(cb.value);
    });

    return {
        requirePosition: document.getElementById('filter-require-position').checked,
        maxHopsAway: document.getElementById('filter-max-hops').value || null,
        lastHeardWithin: document.getElementById('filter-last-heard').value || null,
        roles: roles.length > 0 ? roles : null,
        clearExisting: document.getElementById('filter-clear-existing').checked,
        // Default node settings
        defaultHeight: parseFloat(document.getElementById('filter-default-height').value) || 2.0,
        defaultAntennaGain: parseFloat(document.getElementById('filter-default-gain').value) || 0,
        defaultHopLimit: parseInt(document.getElementById('filter-default-hoplimit').value) || 3
    };
}

function previewNodeDB() {
    const jsonText = document.getElementById('nodedb-json').value.trim();
    if (!jsonText) {
        log('Please paste NodeDB JSON data', 'error');
        return;
    }

    let nodedb;
    try {
        nodedb = JSON.parse(jsonText);
    } catch (e) {
        log('Invalid JSON format: ' + e.message, 'error');
        return;
    }

    const filters = getNodeDBFilters();

    fetch('/api/import/nodedb/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodedb, filters })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            log('Preview error: ' + data.error, 'error');
            return;
        }

        // Update preview stats
        document.getElementById('preview-count').textContent = data.willImport;
        document.getElementById('preview-total').textContent = data.total;
        document.getElementById('preview-no-pos').textContent = data.skipped.no_position || 0;
        document.getElementById('preview-too-far').textContent = data.skipped.too_far || 0;
        document.getElementById('preview-too-old').textContent = data.skipped.too_old || 0;

        // Build preview list
        const listEl = document.getElementById('preview-list');
        listEl.innerHTML = '';

        data.nodes.forEach(node => {
            const item = document.createElement('div');
            item.className = 'preview-node ' + (node.willImport ? 'will-import' : 'will-skip');

            const infoDiv = document.createElement('div');
            const nameSpan = document.createElement('span');
            nameSpan.className = 'node-name';
            nameSpan.textContent = node.longName || node.shortName || node.id;
            infoDiv.appendChild(nameSpan);

            const detailsSpan = document.createElement('span');
            detailsSpan.className = 'node-details';
            const details = [];
            if (node.role) details.push(node.role);
            if (node.hopsAway !== null && node.hopsAway !== undefined) details.push(`${node.hopsAway} hops`);
            if (node.hasPosition) details.push('Has GPS');
            detailsSpan.textContent = details.join(' | ');
            infoDiv.appendChild(document.createElement('br'));
            infoDiv.appendChild(detailsSpan);

            item.appendChild(infoDiv);

            if (!node.willImport && node.skipReason) {
                const reasonSpan = document.createElement('span');
                reasonSpan.className = 'skip-reason';
                const reasons = {
                    'no_position': 'No GPS',
                    'too_far': 'Too far',
                    'too_old': 'Too old',
                    'wrong_role': 'Filtered role'
                };
                reasonSpan.textContent = reasons[node.skipReason] || node.skipReason;
                item.appendChild(reasonSpan);
            }

            listEl.appendChild(item);
        });

        // Show preview section
        document.getElementById('nodedb-preview').classList.remove('hidden');
    })
    .catch(err => {
        log('Preview failed: ' + err.message, 'error');
    });
}

function importNodeDB() {
    const jsonText = document.getElementById('nodedb-json').value.trim();
    if (!jsonText) {
        log('Please paste NodeDB JSON data', 'error');
        return;
    }

    let nodedb;
    try {
        nodedb = JSON.parse(jsonText);
    } catch (e) {
        log('Invalid JSON format: ' + e.message, 'error');
        return;
    }

    const filters = getNodeDBFilters();

    fetch('/api/import/nodedb', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodedb, filters })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            log('Import error: ' + data.error, 'error');
            return;
        }

        // Update map center to the center of imported nodes
        if (data.center && data.center.lat && data.center.lon) {
            state.mapCenter = { lat: data.center.lat, lon: data.center.lon };
        }

        // Close modal and clear form
        document.getElementById('nodedb-modal').classList.add('hidden');
        document.getElementById('nodedb-json').value = '';
        document.getElementById('nodedb-preview').classList.add('hidden');

        // Reload nodes
        loadNodes();

        log(`Imported ${data.imported} nodes from NodeDB`, 'success');
        if (data.skipped && Object.values(data.skipped).some(v => v > 0)) {
            const skippedStr = Object.entries(data.skipped)
                .filter(([k, v]) => v > 0)
                .map(([k, v]) => `${v} ${k.replace('_', ' ')}`)
                .join(', ');
            log(`Skipped: ${skippedStr}`, 'warning');
        }
    })
    .catch(err => {
        log('Import failed: ' + err.message, 'error');
    });
}

const MAX_LOG_ENTRIES = 500;

function log(message, type = 'info') {
    const container = document.getElementById('message-log');
    const timestamp = new Date().toLocaleTimeString();

    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;

    const timestampSpan = document.createElement('span');
    timestampSpan.className = 'timestamp';
    timestampSpan.textContent = `[${timestamp}]`;

    entry.appendChild(timestampSpan);
    entry.appendChild(document.createTextNode(message));

    container.appendChild(entry);

    // Prune old entries to prevent memory bloat
    while (container.children.length > MAX_LOG_ENTRIES) {
        container.removeChild(container.firstChild);
    }

    container.scrollTop = container.scrollHeight;
}

// ============== Network Statistics ==============

function refreshStatistics() {
    fetch('/api/statistics')
        .then(res => res.json())
        .then(data => {
            document.getElementById('stat-total-nodes').textContent = data.totalNodes;
            document.getElementById('stat-connectivity').textContent = data.networkConnectivity + '%';
            document.getElementById('stat-avg-hops').textContent = data.averageHops;
            document.getElementById('stat-diameter').textContent = data.networkDiameter;
            document.getElementById('stat-avg-snr').textContent = data.averageSNR + ' dB';
            document.getElementById('stat-coverage').textContent = data.coverage.area + ' km²';
            document.getElementById('stat-links-total').textContent = data.links.total;

            // Update link quality bars
            const total = data.links.total || 1;
            document.getElementById('stat-links-good').style.width = (data.links.good / total * 100) + '%';
            document.getElementById('stat-links-medium').style.width = (data.links.medium / total * 100) + '%';
            document.getElementById('stat-links-poor').style.width = (data.links.poor / total * 100) + '%';

            // Show isolated nodes warning if any
            const isolatedWarning = document.getElementById('isolated-nodes-warning');
            if (data.isolatedNodes && data.isolatedNodes.length > 0) {
                isolatedWarning.classList.remove('hidden');
                document.getElementById('isolated-nodes-text').textContent =
                    `${data.isolatedNodes.length} isolated node(s): ${data.isolatedNodes.map(n => n.name).join(', ')}`;
            } else {
                isolatedWarning.classList.add('hidden');
            }
        })
        .catch(err => {
            debugLog('Failed to load statistics:', err);
        });
}

// ============== Save/Load Scenarios ==============

let pendingScenario = null;

function saveScenario() {
    const name = document.getElementById('scenario-name').value || 'Untitled Scenario';
    const description = document.getElementById('scenario-description').value || '';

    // Collect current map view state
    const mapView = {
        centerLat: state.mapCenter.lat,
        centerLon: state.mapCenter.lon,
        zoomLevel: state.zoomLevel,
        viewOffsetX: state.viewOffset.x,
        viewOffsetY: state.viewOffset.y
    };

    // Collect channel configuration
    const channels = [{
        name: document.getElementById('channel-name').value || 'Default',
        pskType: document.getElementById('channel-psk-type').value,
        psk: document.getElementById('channel-psk-custom').value || '',
        encryptionEnabled: document.getElementById('channel-encryption-enabled').checked,
        index: 0
    }];

    fetch('/api/scenario/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description, mapView, channels })
    })
    .then(res => res.json())
    .then(scenario => {
        // Create downloadable file
        const blob = new Blob([JSON.stringify(scenario, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${name.replace(/[^a-z0-9]/gi, '_')}.scenario.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        document.getElementById('save-modal').classList.add('hidden');
        log(`Scenario "${name}" saved`, 'success');
    })
    .catch(err => {
        log('Failed to save scenario: ' + err.message, 'error');
    });
}

function previewScenarioFile(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        try {
            const scenario = JSON.parse(event.target.result);
            pendingScenario = scenario;

            // Show preview
            document.getElementById('scenario-preview-name').textContent = scenario.name || 'Untitled';
            document.getElementById('scenario-preview-nodes').textContent = (scenario.nodes || []).length;
            document.getElementById('scenario-preview-date').textContent = scenario.savedAt ?
                new Date(scenario.savedAt).toLocaleString() : 'Unknown';
            document.getElementById('scenario-preview-desc').textContent = scenario.description || 'No description';

            document.getElementById('scenario-preview').classList.remove('hidden');
            document.getElementById('btn-load-confirm').disabled = false;
        } catch (err) {
            log('Invalid scenario file: ' + err.message, 'error');
            pendingScenario = null;
            document.getElementById('btn-load-confirm').disabled = true;
        }
    };
    reader.readAsText(file);
}

function loadScenario() {
    if (!pendingScenario) return;

    fetch('/api/scenario/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pendingScenario)
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            log('Load error: ' + data.error, 'error');
            return;
        }

        // Restore map view
        if (data.mapView) {
            if (data.mapView.centerLat && data.mapView.centerLon) {
                state.mapCenter = { lat: data.mapView.centerLat, lon: data.mapView.centerLon };
            }
            if (data.mapView.zoomLevel) {
                state.zoomLevel = data.mapView.zoomLevel;
            }
            if (data.mapView.viewOffsetX !== undefined) {
                state.viewOffset.x = data.mapView.viewOffsetX;
                state.viewOffset.y = data.mapView.viewOffsetY;
            }
            updateZoomDisplay();
        }

        // Restore channel config
        if (data.channels && data.channels.length > 0) {
            const ch = data.channels[0];
            document.getElementById('channel-name').value = ch.name || 'Default';
            document.getElementById('channel-psk-type').value = ch.pskType || 'default';
            document.getElementById('channel-psk-custom').value = ch.psk || '';
            document.getElementById('channel-encryption-enabled').checked = ch.encryptionEnabled !== false;
            updateEncryptionStatus();
        }

        // Close modal and reload
        document.getElementById('load-modal').classList.add('hidden');
        document.getElementById('scenario-file').value = '';
        pendingScenario = null;

        loadNodes();
        log(`Loaded scenario "${data.name}" with ${data.nodeCount} nodes`, 'success');
    })
    .catch(err => {
        log('Failed to load scenario: ' + err.message, 'error');
    });
}

// ============== Channel & Encryption ==============

function updateEncryptionStatus() {
    const enabled = document.getElementById('channel-encryption-enabled').checked;
    const pskType = document.getElementById('channel-psk-type').value;
    const statusEl = document.getElementById('encryption-status');
    const iconEl = statusEl.querySelector('.status-icon');
    const textEl = statusEl.querySelector('.status-text');

    if (!enabled || pskType === 'none') {
        iconEl.textContent = '\u{1F513}'; // Unlocked
        iconEl.className = 'status-icon unencrypted';
        textEl.textContent = 'Messages are NOT encrypted';
    } else {
        iconEl.textContent = '\u{1F512}'; // Locked
        iconEl.className = 'status-icon encrypted';
        const keyType = pskType === 'default' ? 'default key' :
                       pskType === 'random' ? 'random key' : 'custom key';
        textEl.textContent = `Messages encrypted with ${keyType}`;
    }
}

// ============== GPS Coordinate Support ==============

function addNodeWithGPS() {
    const useGPS = document.getElementById('use-gps-coords').checked;
    let x, y;

    if (useGPS) {
        const lat = parseFloat(document.getElementById('node-lat').value);
        const lon = parseFloat(document.getElementById('node-lon').value);

        if (isNaN(lat) || isNaN(lon)) {
            log('Invalid GPS coordinates', 'error');
            return null;
        }

        // Convert to local meters relative to map center
        const coords = latLonToMeters(lat, lon, state.mapCenter.lat, state.mapCenter.lon);
        x = coords.x;
        y = coords.y;
    } else {
        x = parseFloat(document.getElementById('node-x').value) || 0;
        y = parseFloat(document.getElementById('node-y').value) || 0;
    }

    return { x, y, useGPS };
}

// ============== Performance Optimization ==============

// Tile preloading for smoother map experience
const tilePreloadQueue = [];
let preloadingActive = false;

function preloadTilesAroundView() {
    if (preloadingActive) return;

    const margin = 2; // Preload 2 tiles beyond visible area
    const zoom = state.zoomLevel;
    const mpp = metersPerPixel(state.mapCenter.lat, zoom);

    const centerPixel = latLonToPixel(state.mapCenter.lat, state.mapCenter.lon, zoom);
    const offsetPixels = {
        x: state.viewOffset.x,
        y: state.viewOffset.y
    };

    const topLeftTileX = Math.floor((centerPixel.x + offsetPixels.x - state.canvas.width / 2 - margin * state.tileSize) / state.tileSize);
    const topLeftTileY = Math.floor((centerPixel.y + offsetPixels.y - state.canvas.height / 2 - margin * state.tileSize) / state.tileSize);
    const bottomRightTileX = Math.ceil((centerPixel.x + offsetPixels.x + state.canvas.width / 2 + margin * state.tileSize) / state.tileSize);
    const bottomRightTileY = Math.ceil((centerPixel.y + offsetPixels.y + state.canvas.height / 2 + margin * state.tileSize) / state.tileSize);

    for (let tx = topLeftTileX; tx <= bottomRightTileX; tx++) {
        for (let ty = topLeftTileY; ty <= bottomRightTileY; ty++) {
            const key = `${zoom}/${tx}/${ty}`;
            if (!state.tileCache.has(key) && !state.loadingTiles.has(key)) {
                tilePreloadQueue.push({ zoom, x: tx, y: ty });
            }
        }
    }

    if (tilePreloadQueue.length > 0) {
        preloadingActive = true;
        preloadNextTile();
    }
}

function preloadNextTile() {
    if (tilePreloadQueue.length === 0) {
        preloadingActive = false;
        return;
    }

    const tile = tilePreloadQueue.shift();
    const key = `${tile.zoom}/${tile.x}/${tile.y}`;

    if (state.tileCache.has(key) || state.loadingTiles.has(key)) {
        preloadNextTile();
        return;
    }

    state.loadingTiles.add(key);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
        state.tileCache.set(key, img);
        state.loadingTiles.delete(key);
        setTimeout(preloadNextTile, 50); // Small delay to not overwhelm
    };
    img.onerror = () => {
        state.loadingTiles.delete(key);
        setTimeout(preloadNextTile, 100);
    };
    img.src = `${TILE_SERVER}/${tile.zoom}/${tile.x}/${tile.y}.png`;
}

// Animation frame throttling for large networks
const ANIMATION_THROTTLE_THRESHOLD = 20; // nodes
let lastAnimationFrame = 0;
const MIN_FRAME_INTERVAL = 1000 / 30; // 30 FPS max for large networks

function shouldThrottleAnimation() {
    return state.nodes.size > ANIMATION_THROTTLE_THRESHOLD;
}

function throttledRequestAnimationFrame(callback) {
    if (shouldThrottleAnimation()) {
        const now = performance.now();
        const elapsed = now - lastAnimationFrame;
        if (elapsed < MIN_FRAME_INTERVAL) {
            setTimeout(() => {
                lastAnimationFrame = performance.now();
                requestAnimationFrame(callback);
            }, MIN_FRAME_INTERVAL - elapsed);
            return;
        }
        lastAnimationFrame = now;
    }
    requestAnimationFrame(callback);
}
