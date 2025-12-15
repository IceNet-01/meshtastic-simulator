#!/usr/bin/env python3
"""
Meshtastic Web Simulator
A web-based wrapper for the Meshtasticator interactive simulator.
Provides a user-friendly interface for mesh network planning.
"""

import os
import sys
import json
import time
import math
import logging
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any

import yaml
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# Configure logging - set to WARNING by default, DEBUG for verbose output
logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'WARNING').upper(),
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add the upstream meshtasticator to the path
UPSTREAM_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'upstream-meshtasticator')
sys.path.insert(0, UPSTREAM_PATH)

# Set matplotlib to non-interactive backend before importing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lib.config import Config
from lib import phy

# Import terrain module
try:
    from terrain import check_line_of_sight, meters_to_latlon, clear_elevation_cache
    TERRAIN_AVAILABLE = True
except ImportError:
    TERRAIN_AVAILABLE = False
    logger.warning("Terrain module not available")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'meshtastic-simulator-secret')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
simulator_state = {
    'nodes': {},
    'simulation_running': False,
    'simulation_instance': None,
    'message_log': [],
    'config': None
}

# Message queue for communication between simulator and web interface
message_queue = queue.Queue()


class WebSimulatorConfig(Config):
    """Extended config for web simulator."""

    def __init__(self):
        super().__init__()
        # Smaller default area for web visualization
        self.XSIZE = 10000  # 10km x 10km area
        self.YSIZE = 10000
        # Path loss adjustment in dB (negative values increase range)
        # Default -15 dB gives ~5km range with 2m height instead of ~2km
        # Real-world Meshtastic typically achieves 3-8km in suburban areas
        self.PATH_LOSS_ADJUSTMENT = -15.0

        # Terrain/elevation settings
        self.TERRAIN_ENABLED = False  # Enable terrain-aware LOS calculations
        self.TERRAIN_REF_LAT = 39.8283  # Reference latitude for meter-to-latlon conversion
        self.TERRAIN_REF_LON = -98.5795  # Reference longitude (center of USA)

    def to_dict(self) -> dict:
        """Convert config to JSON-serializable dict."""
        return {
            'model': self.MODEL,
            'xsize': self.XSIZE,
            'ysize': self.YSIZE,
            'hopLimit': self.hopLimit,
            'defaultHeight': self.HM,
            'defaultGain': self.GL,
            'region': 'US',
            'modem': self.MODEM,
            'pathLossAdjustment': self.PATH_LOSS_ADJUSTMENT,
            'terrainEnabled': self.TERRAIN_ENABLED,
            'terrainRefLat': self.TERRAIN_REF_LAT,
            'terrainRefLon': self.TERRAIN_REF_LON,
            'terrainAvailable': TERRAIN_AVAILABLE,
            'pathlossModels': [
                {'id': 0, 'name': 'Log-distance'},
                {'id': 1, 'name': 'Okumura-Hata (small/medium cities)'},
                {'id': 2, 'name': 'Okumura-Hata (metropolitan)'},
                {'id': 3, 'name': 'Okumura-Hata (suburban)'},
                {'id': 4, 'name': 'Okumura-Hata (rural)'},
                {'id': 5, 'name': '3GPP (suburban macro-cell)'},
                {'id': 6, 'name': '3GPP (metropolitan macro-cell)'},
            ],
            'roles': [
                {'id': 'CLIENT', 'name': 'Client'},
                {'id': 'CLIENT_MUTE', 'name': 'Client Mute'},
                {'id': 'ROUTER', 'name': 'Router'},
                {'id': 'REPEATER', 'name': 'Repeater'},
            ]
        }


class SimulatorNode:
    """Represents a node in the simulation."""

    def __init__(self, node_id: int, x: float, y: float, **kwargs):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.z = kwargs.get('z', 1.0)  # height in meters
        self.role = kwargs.get('role', 'CLIENT')
        self.hop_limit = kwargs.get('hop_limit', 3)
        self.antenna_gain = kwargs.get('antenna_gain', 0)
        self.name = kwargs.get('name', f'Node {node_id}')
        self.neighbor_info = kwargs.get('neighbor_info', False)
        # NodeDB import fields
        self.source = kwargs.get('source', 'manual')  # 'manual', 'imported', 'simulated'
        self.meshtastic_id = kwargs.get('meshtastic_id', None)  # Original !hex ID
        self.short_name = kwargs.get('short_name', None)
        self.long_name = kwargs.get('long_name', None)
        self.hw_model = kwargs.get('hw_model', None)
        self.last_heard = kwargs.get('last_heard', None)
        self.snr = kwargs.get('snr', None)
        self.hops_away = kwargs.get('hops_away', None)
        self.latitude = kwargs.get('latitude', None)
        self.longitude = kwargs.get('longitude', None)
        self.battery_level = kwargs.get('battery_level', None)

    def to_dict(self) -> dict:
        result = {
            'id': self.node_id,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'role': self.role,
            'hopLimit': self.hop_limit,
            'antennaGain': self.antenna_gain,
            'name': self.name,
            'neighborInfo': self.neighbor_info,
            'source': self.source
        }
        # Include NodeDB fields if present
        if self.meshtastic_id:
            result['meshtasticId'] = self.meshtastic_id
        if self.short_name:
            result['shortName'] = self.short_name
        if self.long_name:
            result['longName'] = self.long_name
        if self.hw_model:
            result['hwModel'] = self.hw_model
        if self.last_heard:
            result['lastHeard'] = self.last_heard
        if self.snr is not None:
            result['snr'] = self.snr
        if self.hops_away is not None:
            result['hopsAway'] = self.hops_away
        if self.latitude is not None:
            result['latitude'] = self.latitude
        if self.longitude is not None:
            result['longitude'] = self.longitude
        if self.battery_level is not None:
            result['batteryLevel'] = self.battery_level
        return result

    def to_meshtasticator_config(self) -> dict:
        """Convert to format expected by Meshtasticator."""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'isRouter': self.role == 'ROUTER',
            'isRepeater': self.role == 'REPEATER',
            'isClientMute': self.role == 'CLIENT_MUTE',
            'hopLimit': self.hop_limit,
            'antennaGain': self.antenna_gain,
            'neighborInfo': self.neighbor_info
        }


class WebSimulator:
    """Manages the web-based simulation."""

    def __init__(self):
        self.config = WebSimulatorConfig()
        self.nodes: Dict[int, SimulatorNode] = {}
        self.next_node_id = 0
        self.simulation_thread: Optional[threading.Thread] = None
        self.running = False
        self.interactive_sim = None

    def add_node(self, x: float, y: float, **kwargs) -> SimulatorNode:
        """Add a node to the simulation."""
        node = SimulatorNode(self.next_node_id, x, y, **kwargs)
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node

    def update_node(self, node_id: int, **kwargs) -> Optional[SimulatorNode]:
        """Update an existing node."""
        if node_id not in self.nodes:
            return None
        node = self.nodes[node_id]
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
            elif key == 'hopLimit':
                node.hop_limit = value
            elif key == 'antennaGain':
                node.antenna_gain = value
        return node

    def remove_node(self, node_id: int) -> bool:
        """Remove a node from the simulation."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False

    def clear_nodes(self):
        """Remove all nodes."""
        self.nodes.clear()
        self.next_node_id = 0

    def get_node_config_yaml(self) -> str:
        """Generate nodeConfig.yaml content."""
        config = {}
        for i, node in enumerate(self.nodes.values()):
            config[i] = node.to_meshtasticator_config()
        return yaml.dump(config)

    def calculate_coverage(self, node: SimulatorNode) -> float:
        """Calculate estimated coverage radius for a node."""
        return phy.estimate_max_range(node.antenna_gain)

    def calculate_link_quality(self, node1: SimulatorNode, node2: SimulatorNode) -> dict:
        """Calculate link quality between two nodes."""
        dist = np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2 + (node1.z - node2.z)**2)
        base_path_loss = phy.estimate_path_loss(self.config, dist, self.config.FREQ, node1.z, node2.z)
        # Apply path loss adjustment (negative values reduce path loss, increasing range)
        path_loss = base_path_loss + self.config.PATH_LOSS_ADJUSTMENT

        # Terrain-based obstruction loss
        terrain_loss = 0.0
        terrain_info = None
        if self.config.TERRAIN_ENABLED and TERRAIN_AVAILABLE:
            try:
                # Convert node positions (meters) to lat/lon
                lat1, lon1 = meters_to_latlon(node1.x, node1.y,
                                               self.config.TERRAIN_REF_LAT,
                                               self.config.TERRAIN_REF_LON)
                lat2, lon2 = meters_to_latlon(node2.x, node2.y,
                                               self.config.TERRAIN_REF_LAT,
                                               self.config.TERRAIN_REF_LON)

                # Check line of sight with terrain
                los_result = check_line_of_sight(
                    lat1, lon1, node1.z,
                    lat2, lon2, node2.z,
                    freq_mhz=self.config.FREQ / 1e6
                )

                terrain_loss = los_result.get('obstruction_loss', 0)
                # Ensure terrain_loss is a valid number
                if terrain_loss is None or not isinstance(terrain_loss, (int, float)) or math.isnan(terrain_loss) or math.isinf(terrain_loss):
                    terrain_loss = 0.0
                terrain_info = {
                    'hasLos': los_result.get('has_los', True),
                    'obstructionLoss': float(terrain_loss),
                    'clearanceRatio': float(los_result.get('clearance_ratio', 1.0) or 1.0)
                }
            except Exception as e:
                logger.warning(f"Terrain check failed for nodes {node1.id}->{node2.id}: {e}")
                terrain_loss = 0.0

        path_loss += terrain_loss
        rssi = self.config.PTX + node1.antenna_gain - path_loss
        snr = rssi - self.config.NOISE_LEVEL
        can_receive = bool(rssi >= self.config.SENSMODEM[self.config.MODEM])

        # Helper to ensure valid JSON numbers (no NaN/Infinity)
        def safe_float(val, default=0.0):
            if val is None or not isinstance(val, (int, float)):
                return default
            if math.isnan(val) or math.isinf(val):
                return default
            return float(val)

        result = {
            'distance': safe_float(round(dist, 1)),
            'pathLoss': safe_float(round(path_loss, 2)),
            'rssi': safe_float(round(rssi, 2)),
            'snr': safe_float(round(snr, 2)),
            'canReceive': can_receive,
            'signalQuality': int(self._rssi_to_quality(rssi)) if can_receive else 0
        }

        if terrain_info:
            result['terrain'] = terrain_info

        return result

    def _rssi_to_quality(self, rssi: float) -> int:
        """Convert RSSI to signal quality percentage."""
        min_rssi = self.config.SENSMODEM[self.config.MODEM]
        max_rssi = -50  # Good signal
        if rssi >= max_rssi:
            return 100
        if rssi <= min_rssi:
            return 0
        return int(100 * (rssi - min_rssi) / (max_rssi - min_rssi))

    def get_network_topology(self) -> dict:
        """Calculate full network topology with all links."""
        links = []
        for n1_id, n1 in self.nodes.items():
            for n2_id, n2 in self.nodes.items():
                if n1_id < n2_id:  # Avoid duplicates
                    quality = self.calculate_link_quality(n1, n2)
                    if quality['canReceive']:
                        links.append({
                            'source': n1_id,
                            'target': n2_id,
                            **quality
                        })
        return {
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'links': links
        }

    def start_simulation(self, use_docker: bool = True) -> bool:
        """Start the actual Meshtasticator simulation."""
        if self.running:
            return False
        if len(self.nodes) < 2:
            return False

        self.running = True
        self.simulation_thread = threading.Thread(
            target=self._run_simulation,
            args=(use_docker,),
            daemon=True
        )
        self.simulation_thread.start()
        return True

    def _run_simulation(self, use_docker: bool):
        """Run the simulation in a background thread."""
        try:
            # Write node config
            out_dir = os.path.join(UPSTREAM_PATH, 'out')
            os.makedirs(out_dir, exist_ok=True)
            config_path = os.path.join(out_dir, 'nodeConfig.yaml')

            with open(config_path, 'w') as f:
                config = {}
                for i, node in enumerate(self.nodes.values()):
                    config[i] = node.to_meshtasticator_config()
                yaml.dump(config, f)

            socketio.emit('simulation_status', {
                'status': 'starting',
                'message': 'Initializing simulation...'
            })

            # Import and run the interactive sim
            from lib.interactive import InteractiveSim, CommandProcessor
            import argparse

            # Create args namespace
            args = argparse.Namespace(
                script=False,
                docker=use_docker,
                from_file=True,
                forward=False,
                collisions=False,
                program=os.getcwd(),
                nrNodes=0
            )

            socketio.emit('simulation_status', {
                'status': 'running',
                'message': f'Simulation running with {len(self.nodes)} nodes'
            })

            # Note: The actual InteractiveSim requires a display for matplotlib
            # and blocks for user input. For web use, we'd need to modify it
            # or create a non-blocking version. For now, emit that it's ready.
            socketio.emit('simulation_status', {
                'status': 'ready',
                'message': 'Simulation environment ready. Use the command interface to send messages.'
            })

        except Exception as e:
            socketio.emit('simulation_status', {
                'status': 'error',
                'message': str(e)
            })
        finally:
            self.running = False

    def stop_simulation(self):
        """Stop the running simulation."""
        self.running = False
        if self.interactive_sim:
            try:
                self.interactive_sim.close_nodes()
            except:
                pass
            self.interactive_sim = None


# Global simulator instance
simulator = WebSimulator()


# ============== Routes ==============

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/config')
def get_config():
    """Get simulator configuration."""
    return jsonify(simulator.config.to_dict())


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update simulator configuration with validation."""
    data = request.json
    errors = []

    if 'model' in data:
        model = int(data['model'])
        if 0 <= model <= 6:
            simulator.config.MODEL = model
        else:
            errors.append('model must be between 0 and 6')

    if 'xsize' in data:
        xsize = float(data['xsize'])
        if 100 <= xsize <= 1000000:
            simulator.config.XSIZE = xsize
        else:
            errors.append('xsize must be between 100 and 1000000 meters')

    if 'ysize' in data:
        ysize = float(data['ysize'])
        if 100 <= ysize <= 1000000:
            simulator.config.YSIZE = ysize
        else:
            errors.append('ysize must be between 100 and 1000000 meters')

    if 'hopLimit' in data:
        hop_limit = int(data['hopLimit'])
        if 1 <= hop_limit <= 7:
            simulator.config.hopLimit = hop_limit
        else:
            errors.append('hopLimit must be between 1 and 7')

    if 'defaultHeight' in data:
        height = float(data['defaultHeight'])
        if 0 <= height <= 1000:
            simulator.config.HM = height
        else:
            errors.append('defaultHeight must be between 0 and 1000 meters')

    if 'defaultGain' in data:
        gain = float(data['defaultGain'])
        if -20 <= gain <= 30:
            simulator.config.GL = gain
        else:
            errors.append('defaultGain must be between -20 and 30 dBi')

    if 'pathLossAdjustment' in data:
        adjustment = float(data['pathLossAdjustment'])
        if -30 <= adjustment <= 30:
            simulator.config.PATH_LOSS_ADJUSTMENT = adjustment
        else:
            errors.append('pathLossAdjustment must be between -30 and 30 dB')

    if 'terrainEnabled' in data:
        simulator.config.TERRAIN_ENABLED = bool(data['terrainEnabled'])
        if simulator.config.TERRAIN_ENABLED and TERRAIN_AVAILABLE:
            clear_elevation_cache()  # Clear cache when enabling

    if 'terrainRefLat' in data:
        lat = float(data['terrainRefLat'])
        if -90 <= lat <= 90:
            simulator.config.TERRAIN_REF_LAT = lat
        else:
            errors.append('terrainRefLat must be between -90 and 90')

    if 'terrainRefLon' in data:
        lon = float(data['terrainRefLon'])
        if -180 <= lon <= 180:
            simulator.config.TERRAIN_REF_LON = lon
        else:
            errors.append('terrainRefLon must be between -180 and 180')

    if errors:
        return jsonify({'status': 'error', 'errors': errors, 'config': simulator.config.to_dict()}), 400

    return jsonify({'status': 'ok', 'config': simulator.config.to_dict()})


@app.route('/api/nodes')
def get_nodes():
    """Get all nodes."""
    return jsonify([n.to_dict() for n in simulator.nodes.values()])


@app.route('/api/nodes', methods=['POST'])
def add_node():
    """Add a new node."""
    data = request.json
    node = simulator.add_node(
        x=float(data.get('x', 0)),
        y=float(data.get('y', 0)),
        z=float(data.get('z', simulator.config.HM)),
        role=data.get('role', 'CLIENT'),
        hop_limit=int(data.get('hopLimit', simulator.config.hopLimit)),
        antenna_gain=float(data.get('antennaGain', simulator.config.GL)),
        name=data.get('name', f'Node {simulator.next_node_id}')
    )

    # Calculate coverage radius
    coverage = simulator.calculate_coverage(node)

    # Emit to all clients
    socketio.emit('node_added', {
        **node.to_dict(),
        'coverageRadius': coverage
    })

    return jsonify({
        **node.to_dict(),
        'coverageRadius': coverage
    })


@app.route('/api/nodes/<int:node_id>', methods=['PUT'])
def update_node(node_id):
    """Update an existing node."""
    data = request.json
    node = simulator.update_node(node_id, **data)
    if node:
        coverage = simulator.calculate_coverage(node)
        socketio.emit('node_updated', {
            **node.to_dict(),
            'coverageRadius': coverage
        })
        return jsonify({
            **node.to_dict(),
            'coverageRadius': coverage
        })
    return jsonify({'error': 'Node not found'}), 404


@app.route('/api/nodes/<int:node_id>', methods=['DELETE'])
def delete_node(node_id):
    """Delete a node."""
    if simulator.remove_node(node_id):
        socketio.emit('node_removed', {'id': node_id})
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'Node not found'}), 404


@app.route('/api/nodes/clear', methods=['POST'])
def clear_nodes():
    """Clear all nodes."""
    simulator.clear_nodes()
    socketio.emit('nodes_cleared')
    return jsonify({'status': 'ok'})


@app.route('/api/topology')
def get_topology():
    """Get full network topology."""
    return jsonify(simulator.get_network_topology())


@app.route('/api/link/<int:node1_id>/<int:node2_id>')
def get_link_quality(node1_id, node2_id):
    """Get link quality between two nodes."""
    if node1_id not in simulator.nodes or node2_id not in simulator.nodes:
        return jsonify({'error': 'Node not found'}), 404

    n1 = simulator.nodes[node1_id]
    n2 = simulator.nodes[node2_id]
    return jsonify(simulator.calculate_link_quality(n1, n2))


@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start the simulation."""
    data = request.json or {}
    use_docker = data.get('useDocker', True)

    if len(simulator.nodes) < 2:
        return jsonify({'error': 'Need at least 2 nodes'}), 400

    if simulator.start_simulation(use_docker=use_docker):
        return jsonify({'status': 'starting'})
    return jsonify({'error': 'Simulation already running'}), 400


@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop the simulation."""
    simulator.stop_simulation()
    return jsonify({'status': 'stopped'})


@app.route('/api/simulation/status')
def get_simulation_status():
    """Get simulation status."""
    return jsonify({
        'running': simulator.running,
        'nodeCount': len(simulator.nodes)
    })


@app.route('/api/statistics')
def get_statistics():
    """Get network statistics dashboard data."""
    nodes = list(simulator.nodes.values())
    if not nodes:
        return jsonify({
            'totalNodes': 0,
            'nodesByRole': {},
            'networkConnectivity': 0,
            'averageHops': 0,
            'coverage': {'area': 0, 'radius': 0},
            'links': {'total': 0, 'good': 0, 'medium': 0, 'poor': 0},
            'isolatedNodes': [],
            'networkDiameter': 0,
            'averageSNR': 0
        })

    # Count nodes by role
    nodes_by_role = {}
    for node in nodes:
        role = node.role
        nodes_by_role[role] = nodes_by_role.get(role, 0) + 1

    # Calculate all links and statistics
    links = []
    all_snrs = []
    connectivity_matrix = {n.node_id: set() for n in nodes}

    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i < j:
                quality = simulator.calculate_link_quality(n1, n2)
                if quality['canReceive']:
                    links.append(quality)
                    all_snrs.append(quality['snr'])
                    connectivity_matrix[n1.node_id].add(n2.node_id)
                    connectivity_matrix[n2.node_id].add(n1.node_id)

    # Link quality distribution
    good_links = sum(1 for l in links if l['signalQuality'] >= 70)
    medium_links = sum(1 for l in links if 40 <= l['signalQuality'] < 70)
    poor_links = sum(1 for l in links if l['signalQuality'] < 40)

    # Find isolated nodes (no connections)
    isolated_nodes = [
        {'id': nid, 'name': simulator.nodes[nid].name or f'Node {nid}'}
        for nid, connections in connectivity_matrix.items()
        if len(connections) == 0
    ]

    # Calculate network connectivity (% of max possible connections)
    max_connections = len(nodes) * (len(nodes) - 1) / 2 if len(nodes) > 1 else 1
    connectivity = (len(links) / max_connections * 100) if max_connections > 0 else 0

    # Calculate network diameter using BFS
    def bfs_shortest_paths(start_id):
        visited = {start_id: 0}
        queue = [start_id]
        while queue:
            current = queue.pop(0)
            for neighbor in connectivity_matrix[current]:
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
        return visited

    max_distance = 0
    total_hops = 0
    hop_count = 0
    for node in nodes:
        distances = bfs_shortest_paths(node.node_id)
        if distances:
            node_max = max(distances.values()) if distances.values() else 0
            max_distance = max(max_distance, node_max)
            for d in distances.values():
                if d > 0:
                    total_hops += d
                    hop_count += 1

    avg_hops = total_hops / hop_count if hop_count > 0 else 0

    # Calculate coverage area (convex hull approximation using bounding box)
    if nodes:
        xs = [n.x for n in nodes]
        ys = [n.y for n in nodes]
        width = max(xs) - min(xs) if xs else 0
        height = max(ys) - min(ys) if ys else 0
        coverage_area = width * height / 1e6  # km²

        # Average coverage radius per node
        avg_coverage_radius = sum(simulator.calculate_coverage(n) for n in nodes) / len(nodes)
    else:
        coverage_area = 0
        avg_coverage_radius = 0

    return jsonify({
        'totalNodes': len(nodes),
        'nodesByRole': nodes_by_role,
        'networkConnectivity': round(connectivity, 1),
        'averageHops': round(avg_hops, 2),
        'coverage': {
            'area': round(coverage_area, 2),
            'radius': round(avg_coverage_radius, 0)
        },
        'links': {
            'total': len(links),
            'good': good_links,
            'medium': medium_links,
            'poor': poor_links
        },
        'isolatedNodes': isolated_nodes,
        'networkDiameter': max_distance,
        'averageSNR': round(sum(all_snrs) / len(all_snrs), 1) if all_snrs else 0
    })


@app.route('/api/export/yaml')
def export_yaml():
    """Export node configuration as YAML."""
    yaml_content = simulator.get_node_config_yaml()
    return yaml_content, 200, {'Content-Type': 'text/yaml'}


@app.route('/api/scenario/save', methods=['POST'])
def save_scenario():
    """Save complete simulation scenario (nodes + settings + map position)."""
    data = request.json or {}
    scenario = {
        'version': '1.0',
        'savedAt': datetime.now().isoformat(),
        'name': data.get('name', 'Untitled Scenario'),
        'description': data.get('description', ''),
        'config': {
            'model': simulator.config.MODEL,
            'xsize': simulator.config.XSIZE,
            'ysize': simulator.config.YSIZE,
            'hopLimit': simulator.config.hopLimit,
            'modem': simulator.config.MODEM
        },
        'mapView': data.get('mapView', {}),
        'nodes': [node.to_dict() for node in simulator.nodes.values()],
        'channels': data.get('channels', [{'name': 'Default', 'psk': 'AQ==', 'index': 0}])
    }
    return jsonify(scenario)


@app.route('/api/scenario/load', methods=['POST'])
def load_scenario():
    """Load a complete simulation scenario."""
    try:
        scenario = request.json
        if not scenario:
            return jsonify({'error': 'No scenario data provided'}), 400

        # Validate version
        version = scenario.get('version', '1.0')

        # Load config
        config_data = scenario.get('config', {})
        if 'model' in config_data:
            simulator.config.MODEL = int(config_data['model'])
        if 'xsize' in config_data:
            simulator.config.XSIZE = float(config_data['xsize'])
        if 'ysize' in config_data:
            simulator.config.YSIZE = float(config_data['ysize'])
        if 'hopLimit' in config_data:
            simulator.config.hopLimit = int(config_data['hopLimit'])
        if 'modem' in config_data:
            simulator.config.MODEM = int(config_data['modem'])

        # Load nodes
        simulator.clear_nodes()
        nodes_data = scenario.get('nodes', [])
        for node_data in nodes_data:
            simulator.add_node(
                x=float(node_data.get('x', 0)),
                y=float(node_data.get('y', 0)),
                z=float(node_data.get('z', 1.0)),
                role=node_data.get('role', 'CLIENT'),
                hop_limit=int(node_data.get('hopLimit', 3)),
                antenna_gain=float(node_data.get('antennaGain', 0)),
                name=node_data.get('name'),
                source=node_data.get('source', 'manual'),
                meshtastic_id=node_data.get('meshtasticId'),
                short_name=node_data.get('shortName'),
                long_name=node_data.get('longName'),
                hw_model=node_data.get('hwModel'),
                latitude=node_data.get('latitude'),
                longitude=node_data.get('longitude')
            )

        # Return map view for frontend to restore
        map_view = scenario.get('mapView', {})
        channels = scenario.get('channels', [])

        socketio.emit('scenario_loaded', {
            'name': scenario.get('name', 'Untitled'),
            'nodeCount': len(simulator.nodes),
            'mapView': map_view,
            'channels': channels
        })

        return jsonify({
            'status': 'ok',
            'name': scenario.get('name', 'Untitled'),
            'nodeCount': len(simulator.nodes),
            'mapView': map_view,
            'channels': channels
        })

    except Exception as e:
        logger.exception("Scenario load error")
        return jsonify({'error': str(e)}), 400


@app.route('/api/import/yaml', methods=['POST'])
def import_yaml():
    """Import node configuration from YAML."""
    try:
        if request.is_json:
            yaml_content = request.json.get('yaml', '')
        else:
            yaml_content = request.data.decode('utf-8')

        config = yaml.safe_load(yaml_content)
        simulator.clear_nodes()

        for node_id, node_config in config.items():
            # Validate required fields
            if 'x' not in node_config or 'y' not in node_config:
                return jsonify({'error': f'Node {node_id} missing required x or y coordinate'}), 400

            role = 'CLIENT'
            if node_config.get('isRouter'):
                role = 'ROUTER'
            elif node_config.get('isRepeater'):
                role = 'REPEATER'
            elif node_config.get('isClientMute'):
                role = 'CLIENT_MUTE'

            simulator.add_node(
                x=float(node_config['x']),
                y=float(node_config['y']),
                z=float(node_config.get('z', 1.0)),
                role=role,
                hop_limit=int(node_config.get('hopLimit', 3)),
                antenna_gain=float(node_config.get('antennaGain', 0)),
                neighbor_info=node_config.get('neighborInfo', False)
            )

        socketio.emit('config_imported', {'nodeCount': len(simulator.nodes)})
        return jsonify({'status': 'ok', 'nodeCount': len(simulator.nodes)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/import/nodedb', methods=['POST'])
def import_nodedb():
    """
    Import nodes from a Meshtastic NodeDB JSON export.

    Expected format (from meshtastic python library):
    {
        "nodes": {
            "!12345678": {
                "num": 12345678,
                "user": {
                    "id": "!12345678",
                    "longName": "Node Name",
                    "shortName": "ND",
                    "hwModel": "TBEAM",
                    "role": "ROUTER"
                },
                "position": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "altitude": 10
                },
                "lastHeard": 1702500000,
                "snr": 10.5,
                "hopsAway": 1,
                "deviceMetrics": {
                    "batteryLevel": 85
                }
            }
        }
    }

    Filters (optional in request):
    - requirePosition: bool - only import nodes with valid lat/lon
    - maxHopsAway: int - only nodes within N hops (null = heard directly by mesh)
    - lastHeardWithin: int - only nodes heard within N seconds
    - roles: list - only nodes with specific roles ['ROUTER', 'CLIENT', etc.]
    - clearExisting: bool - clear existing nodes before import (default: true)
    """
    try:
        data = request.json
        nodedb = data.get('nodedb', data)  # Support both wrapped and raw format
        filters = data.get('filters', {})

        # Parse the nodes - handle different formats
        raw_nodes = {}
        if 'nodes' in nodedb:
            raw_nodes = nodedb['nodes']
        elif isinstance(nodedb, dict):
            # Check if it's a direct node dict (keys are node IDs)
            for key, val in nodedb.items():
                if isinstance(val, dict) and ('user' in val or 'num' in val):
                    raw_nodes = nodedb
                    break

        if not raw_nodes:
            return jsonify({'error': 'No valid nodes found in NodeDB data'}), 400

        # Apply filters
        require_position = filters.get('requirePosition', True)
        max_hops = filters.get('maxHopsAway', None)
        last_heard_within = filters.get('lastHeardWithin', None)
        allowed_roles = filters.get('roles', None)
        clear_existing = filters.get('clearExisting', True)

        # Default node settings (overridable via filters)
        default_height = float(filters.get('defaultHeight', 2.0))  # Default 2m height
        default_antenna_gain = float(filters.get('defaultAntennaGain', 2.0))  # 2 dBi default (stock antenna)
        default_hop_limit = int(filters.get('defaultHopLimit', 3))

        current_time = time.time()

        # Filter and collect valid nodes
        valid_nodes = []
        skipped = {'no_position': 0, 'too_far': 0, 'too_old': 0, 'wrong_role': 0}

        for node_id, node_data in raw_nodes.items():
            # Extract position
            position = node_data.get('position', {})
            lat = position.get('latitude') or position.get('latitudeI', 0) / 1e7
            lon = position.get('longitude') or position.get('longitudeI', 0) / 1e7
            alt = position.get('altitude', 1)

            # Filter: require position
            if require_position and (lat == 0 or lon == 0):
                skipped['no_position'] += 1
                continue

            # Filter: hops away
            hops_away = node_data.get('hopsAway')
            if max_hops is not None and hops_away is not None and hops_away > max_hops:
                skipped['too_far'] += 1
                continue

            # Filter: last heard
            last_heard = node_data.get('lastHeard', 0)
            if last_heard_within is not None and last_heard > 0:
                if current_time - last_heard > last_heard_within:
                    skipped['too_old'] += 1
                    continue

            # Extract user info
            user = node_data.get('user', {})
            role = user.get('role', 'CLIENT')

            # Filter: roles
            if allowed_roles and role not in allowed_roles:
                skipped['wrong_role'] += 1
                continue

            # Extract device metrics
            device_metrics = node_data.get('deviceMetrics', {})

            valid_nodes.append({
                'meshtastic_id': user.get('id') or node_id,
                'num': node_data.get('num'),
                'short_name': user.get('shortName'),
                'long_name': user.get('longName'),
                'hw_model': user.get('hwModel'),
                'role': role,
                'latitude': lat,
                'longitude': lon,
                'altitude': alt,
                'last_heard': last_heard,
                'snr': node_data.get('snr'),
                'hops_away': hops_away,
                'battery_level': device_metrics.get('batteryLevel')
            })

        if not valid_nodes:
            return jsonify({
                'error': 'No nodes passed filters',
                'skipped': skipped
            }), 400

        # Calculate center point and convert to local coordinates
        lats = [n['latitude'] for n in valid_nodes if n['latitude']]
        lons = [n['longitude'] for n in valid_nodes if n['longitude']]

        if not lats or not lons:
            return jsonify({'error': 'No nodes with valid positions'}), 400

        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        # Convert lat/lon to local meter coordinates
        METERS_PER_DEGREE_LAT = 111320
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * abs(math.cos(math.radians(center_lat)))

        # Clear existing nodes if requested
        if clear_existing:
            simulator.clear_nodes()

        # Import nodes
        imported_count = 0
        for node_data in valid_nodes:
            # Convert to local coordinates (meters from center)
            x = (node_data['longitude'] - center_lon) * meters_per_degree_lon
            y = (node_data['latitude'] - center_lat) * METERS_PER_DEGREE_LAT

            # Determine display name
            name = node_data['long_name'] or node_data['short_name'] or node_data['meshtastic_id']

            # Map Meshtastic roles to simulator roles
            role_map = {
                'CLIENT': 'CLIENT',
                'CLIENT_MUTE': 'CLIENT_MUTE',
                'CLIENT_HIDDEN': 'CLIENT_MUTE',
                'ROUTER': 'ROUTER',
                'ROUTER_CLIENT': 'ROUTER',
                'REPEATER': 'REPEATER',
                'TRACKER': 'CLIENT',
                'SENSOR': 'CLIENT',
                'TAK': 'CLIENT',
                'TAK_TRACKER': 'CLIENT',
                'LOST_AND_FOUND': 'CLIENT',
            }
            sim_role = role_map.get(node_data['role'], 'CLIENT')

            # Always use default height for imported nodes
            # NodeDB altitude is MSL (meters above sea level), not AGL (above ground level)
            # For radio propagation, we need AGL which is typically 1-10m for portable devices
            node_height = default_height

            simulator.add_node(
                x=x,
                y=y,
                z=node_height,
                role=sim_role,
                hop_limit=default_hop_limit,
                antenna_gain=default_antenna_gain,
                name=name,
                source='imported',
                meshtastic_id=node_data['meshtastic_id'],
                short_name=node_data['short_name'],
                long_name=node_data['long_name'],
                hw_model=node_data['hw_model'],
                last_heard=node_data['last_heard'],
                snr=node_data['snr'],
                hops_away=node_data['hops_away'],
                latitude=node_data['latitude'],
                longitude=node_data['longitude'],
                battery_level=node_data['battery_level']
            )
            imported_count += 1

        # Update simulator area to fit imported nodes
        if imported_count > 0:
            all_x = [n.x for n in simulator.nodes.values()]
            all_y = [n.y for n in simulator.nodes.values()]
            margin = 1000  # 1km margin
            simulator.config.XSIZE = max(10000, (max(all_x) - min(all_x)) + margin * 2)
            simulator.config.YSIZE = max(10000, (max(all_y) - min(all_y)) + margin * 2)

        socketio.emit('config_imported', {
            'nodeCount': len(simulator.nodes),
            'centerLat': center_lat,
            'centerLon': center_lon
        })

        return jsonify({
            'status': 'ok',
            'imported': imported_count,
            'skipped': skipped,
            'center': {'lat': center_lat, 'lon': center_lon},
            'nodeCount': len(simulator.nodes)
        })

    except Exception as e:
        logger.exception("NodeDB import error")
        return jsonify({'error': str(e)}), 400


@app.route('/api/import/nodedb/preview', methods=['POST'])
def preview_nodedb():
    """
    Preview NodeDB import without actually importing.
    Returns list of nodes that would be imported with current filters.
    """
    try:
        data = request.json
        nodedb = data.get('nodedb', data)
        filters = data.get('filters', {})

        raw_nodes = {}
        if 'nodes' in nodedb:
            raw_nodes = nodedb['nodes']
        elif isinstance(nodedb, dict):
            for key, val in nodedb.items():
                if isinstance(val, dict) and ('user' in val or 'num' in val):
                    raw_nodes = nodedb
                    break

        require_position = filters.get('requirePosition', True)
        max_hops = filters.get('maxHopsAway', None)
        last_heard_within = filters.get('lastHeardWithin', None)
        allowed_roles = filters.get('roles', None)

        current_time = time.time()
        preview_nodes = []
        skipped = {'no_position': 0, 'too_far': 0, 'too_old': 0, 'wrong_role': 0}

        for node_id, node_data in raw_nodes.items():
            position = node_data.get('position', {})
            lat = position.get('latitude') or position.get('latitudeI', 0) / 1e7
            lon = position.get('longitude') or position.get('longitudeI', 0) / 1e7

            user = node_data.get('user', {})
            role = user.get('role', 'CLIENT')
            hops_away = node_data.get('hopsAway')
            last_heard = node_data.get('lastHeard', 0)

            # Apply filters and track why nodes are skipped
            skip_reason = None
            if require_position and (lat == 0 or lon == 0):
                skip_reason = 'no_position'
            elif max_hops is not None and hops_away is not None and hops_away > max_hops:
                skip_reason = 'too_far'
            elif last_heard_within is not None and last_heard > 0 and current_time - last_heard > last_heard_within:
                skip_reason = 'too_old'
            elif allowed_roles and role not in allowed_roles:
                skip_reason = 'wrong_role'

            node_preview = {
                'id': user.get('id') or node_id,
                'shortName': user.get('shortName'),
                'longName': user.get('longName'),
                'role': role,
                'hwModel': user.get('hwModel'),
                'hasPosition': lat != 0 and lon != 0,
                'latitude': lat if lat != 0 else None,
                'longitude': lon if lon != 0 else None,
                'hopsAway': hops_away,
                'lastHeard': last_heard,
                'snr': node_data.get('snr'),
                'willImport': skip_reason is None,
                'skipReason': skip_reason
            }
            preview_nodes.append(node_preview)

            if skip_reason:
                skipped[skip_reason] += 1

        # Sort: importing nodes first, then by name
        preview_nodes.sort(key=lambda n: (not n['willImport'], n.get('longName') or n.get('shortName') or n['id']))

        return jsonify({
            'total': len(preview_nodes),
            'willImport': sum(1 for n in preview_nodes if n['willImport']),
            'skipped': skipped,
            'nodes': preview_nodes
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ============== WebSocket Events ==============

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {
        'nodeCount': len(simulator.nodes),
        'simulationRunning': simulator.running
    })


@socketio.on('request_topology')
def handle_topology_request():
    """Handle topology request."""
    emit('topology', simulator.get_network_topology())


@socketio.on('send_command')
def handle_command(data):
    """Handle simulator commands from web interface."""
    command = data.get('command', '')
    args = data.get('args', {})

    if command == 'broadcast':
        from_node = args.get('from')
        text = args.get('text', 'Test message')
        hop_limit = args.get('hopLimit', 3)

        logger.debug(f"[BROADCAST] From node {from_node}, hop_limit={hop_limit}")

        # Simulate broadcast propagation
        result = simulate_broadcast(from_node, hop_limit)
        logger.debug(f"[BROADCAST] Result: received={result.get('totalReceived')}/{result.get('totalNodes')-1}, maxHopsUsed={result.get('maxHopsUsed')} (limit was {hop_limit})")
        logger.debug(f"[BROADCAST] Propagation steps: {len(result.get('propagation', []))}")
        for step in result.get('propagation', []):
            logger.debug(f"  Hop {step['hop']}: {len(step['transmissions'])} transmissions")
        # Show which nodes received at which hop
        for n in result.get('nodes', []):
            if n['status'] == 'received':
                logger.debug(f"  Node {n['id']} received at hop {n['hop']} (remaining: {n.get('remaining', '?')})")

        msg = f'Broadcast from Node {from_node} (hop limit {hop_limit}): {result["totalReceived"]}/{result["totalNodes"]-1} nodes received'
        if result.get('maxHopsUsed', 0) > 0:
            msg += f', farthest at hop {result["maxHopsUsed"]}'

        emit('command_response', {
            'command': 'broadcast',
            'status': 'success' if result['totalReceived'] > 0 else 'partial',
            'message': msg,
            'simulation': result
        }, broadcast=True)

    elif command == 'dm':
        from_node = args.get('from')
        to_node = args.get('to')
        text = args.get('text', 'Test message')

        # Get hop limit from source node
        source_node = simulator.nodes.get(from_node)
        hop_limit = source_node.hop_limit if source_node else 3

        logger.debug(f"[DM] From node {from_node} to {to_node}, hop_limit={hop_limit}")

        # Simulate direct message routing
        result = simulate_direct_message(from_node, to_node)

        status = 'success' if result['delivered'] else 'failed'
        msg = f'DM {from_node} → {to_node} (hop limit {hop_limit}): {"Delivered" if result["delivered"] else "Failed"}'
        if result['delivered']:
            msg += f' via {len(result["path"])-1} hop(s)'
        else:
            msg += f' - {result.get("reason", "unknown reason")}'

        emit('command_response', {
            'command': 'dm',
            'status': status,
            'message': msg,
            'simulation': result
        }, broadcast=True)

    elif command == 'traceroute':
        from_node = args.get('from')
        to_node = args.get('to')

        # Get hop limit from source node
        source_node = simulator.nodes.get(from_node)
        hop_limit = source_node.hop_limit if source_node else 3

        logger.debug(f"[TRACEROUTE] From node {from_node} to {to_node}, hop_limit={hop_limit}")

        # Simulate traceroute with detailed hop info
        result = simulate_traceroute(from_node, to_node)

        if result['reachable']:
            hop_info = ' → '.join([f"{h['node']}({h['rssi']:.0f}dBm)" for h in result['hops']])
            msg = f'Traceroute {from_node} → {to_node} (hop limit {hop_limit}): {hop_info}'
        else:
            msg = f'Traceroute {from_node} → {to_node} (hop limit {hop_limit}): {result.get("reason", "No route found")}'

        emit('command_response', {
            'command': 'traceroute',
            'status': 'success' if result['reachable'] else 'failed',
            'route': result['path'],
            'message': msg,
            'simulation': result
        }, broadcast=True)

    else:
        emit('command_response', {
            'command': command,
            'status': 'unknown',
            'message': f'Unknown command: {command}'
        })


def simulate_broadcast(from_node: int, hop_limit: int = 3) -> dict:
    """Simulate a broadcast message propagating through the mesh.

    The hop_limit is the ORIGINAL hop limit set by the source node.
    Each hop decrements the remaining hops. When remaining hops = 0,
    nodes receive but don't rebroadcast.
    """
    if from_node not in simulator.nodes:
        return {'error': 'Source node not found', 'totalReceived': 0, 'totalNodes': 0, 'propagation': []}

    source = simulator.nodes[from_node]
    all_nodes = list(simulator.nodes.keys())

    # Track which nodes received the message, at which hop, and remaining hops
    # remaining_hops = hop_limit - hops_traveled
    received = {from_node: {'hop': 0, 'rssi': 0, 'from': None, 'remaining': hop_limit}}
    propagation = []  # List of propagation steps for animation

    # Frontier contains (node_id, remaining_hops) - remaining hops AFTER this node received
    frontier = [(from_node, hop_limit)]
    current_hop = 0

    while frontier and current_hop < 10:  # Safety limit
        next_frontier = []
        hop_transmissions = []
        transmitting_nodes = set()  # Track all nodes that transmit this hop

        for tx_node_id, remaining_hops in frontier:
            # If remaining hops is 0, this node received but won't rebroadcast
            if remaining_hops <= 0:
                continue

            tx_node = simulator.nodes[tx_node_id]
            transmitting_nodes.add(tx_node_id)  # This node is transmitting

            # Check which nodes can hear this transmission
            for rx_node_id, rx_node in simulator.nodes.items():
                if rx_node_id == tx_node_id:
                    continue
                if rx_node_id in received:
                    continue  # Already received

                # Check if can receive
                link = simulator.calculate_link_quality(tx_node, rx_node)
                if link['canReceive']:
                    # Remaining hops for this received message is one less than transmitter had
                    new_remaining = remaining_hops - 1

                    received[rx_node_id] = {
                        'hop': current_hop + 1,
                        'rssi': link['rssi'],
                        'snr': link['snr'],
                        'from': tx_node_id,
                        'remaining': new_remaining
                    }
                    hop_transmissions.append({
                        'from': tx_node_id,
                        'to': rx_node_id,
                        'rssi': float(link['rssi']),
                        'snr': float(link['snr']),
                        'distance': float(link['distance']),
                        'remainingHops': new_remaining
                    })

                    # Only add to frontier if:
                    # 1. Role allows rebroadcast (not CLIENT_MUTE)
                    # 2. There are remaining hops to forward
                    role = rx_node.role
                    if role in ['ROUTER', 'REPEATER', 'CLIENT'] and new_remaining > 0:
                        next_frontier.append((rx_node_id, new_remaining))

        # Record this hop if any nodes transmitted (even if no new receivers)
        if transmitting_nodes:
            propagation.append({
                'hop': current_hop + 1,
                'transmissions': hop_transmissions,
                'transmitters': list(transmitting_nodes)  # All nodes that transmitted this hop
            })

        frontier = next_frontier
        current_hop += 1

        # Stop if no more nodes to process
        if not frontier:
            break

    # Build result
    node_results = []
    for node_id in all_nodes:
        if node_id == from_node:
            node_results.append({'id': node_id, 'status': 'source', 'hop': 0, 'remaining': hop_limit})
        elif node_id in received:
            r = received[node_id]
            node_results.append({
                'id': node_id,
                'status': 'received',
                'hop': r['hop'],
                'rssi': float(r['rssi']),
                'snr': float(r['snr']),
                'from': r['from'],
                'remaining': r.get('remaining', 0)
            })
        else:
            node_results.append({'id': node_id, 'status': 'unreached', 'hop': -1})

    # Count actual propagation hops (should be <= hop_limit)
    max_hop = max((r['hop'] for r in received.values()), default=0)

    return {
        'source': from_node,
        'hopLimit': hop_limit,
        'maxHopsUsed': max_hop,
        'totalNodes': len(all_nodes),
        'totalReceived': len(received) - 1,  # Exclude source
        'nodes': node_results,
        'propagation': propagation
    }


def simulate_direct_message(from_node: int, to_node: int) -> dict:
    """Simulate a direct message using Meshtastic flooding.

    The message floods out in all directions (like broadcast) but we track
    whether it reaches the specific destination within the hop limit.
    """
    if from_node not in simulator.nodes or to_node not in simulator.nodes:
        return {
            'delivered': False,
            'path': [],
            'hops': [],
            'propagation': [],
            'reason': 'Node not found'
        }

    # Get hop limit from source node
    hop_limit = simulator.nodes[from_node].hop_limit
    logger.debug(f"[DM FLOOD] Starting flood from node {from_node} with hop_limit={hop_limit}")

    # Use flood simulation (same as broadcast) to see how message propagates
    flood_result = simulate_broadcast(from_node, hop_limit)
    logger.debug(f"[DM FLOOD] Flood result: {len(flood_result.get('nodes', []))} nodes reached, {len(flood_result.get('propagation', []))} propagation hops")

    # Check if destination was reached in the flood
    destination_reached = False
    destination_hop = -1
    for node_info in flood_result.get('nodes', []):
        if node_info['id'] == to_node and node_info['status'] == 'received':
            destination_reached = True
            destination_hop = node_info['hop']
            break
    logger.debug(f"[DM FLOOD] Destination {to_node} reached: {destination_reached}, at hop: {destination_hop}")

    # If destination was reached, trace back the path
    path = []
    hops = []
    if destination_reached:
        # Reconstruct path from propagation data
        path = reconstruct_path(flood_result, from_node, to_node)

        # Build hop details for the path
        for i in range(len(path) - 1):
            n1 = simulator.nodes[path[i]]
            n2 = simulator.nodes[path[i + 1]]
            link = simulator.calculate_link_quality(n1, n2)
            hops.append({
                'from': path[i],
                'to': path[i + 1],
                'rssi': float(link['rssi']),
                'snr': float(link['snr']),
                'distance': float(link['distance']),
                'quality': int(link['signalQuality'])
            })

    return {
        'delivered': destination_reached,
        'source': from_node,
        'destination': to_node,
        'path': path,
        'hops': hops,
        'totalHops': len(path) - 1 if path else 0,
        'propagation': flood_result.get('propagation', []),
        'floodResult': flood_result,
        'hopLimit': hop_limit,
        'reason': None if destination_reached else f'Message died after {hop_limit} hops - destination not reached'
    }


def reconstruct_path(flood_result: dict, from_node: int, to_node: int) -> List[int]:
    """Reconstruct the path a message took from source to destination."""
    # Build a map of node -> who it received from
    received_from = {}
    for node_info in flood_result.get('nodes', []):
        if node_info.get('from') is not None:
            received_from[node_info['id']] = node_info['from']

    # Trace back from destination to source
    path = [to_node]
    current = to_node
    while current != from_node and current in received_from:
        current = received_from[current]
        path.insert(0, current)

    # Verify we reached the source
    if path[0] != from_node:
        return []

    return path


def simulate_traceroute(from_node: int, to_node: int) -> dict:
    """Simulate traceroute using Meshtastic flooding.

    Like DM, traceroute floods out and we track where it reaches.
    Shows the full propagation even if destination isn't reached.
    """
    if from_node not in simulator.nodes or to_node not in simulator.nodes:
        return {
            'reachable': False,
            'path': [],
            'hops': [],
            'propagation': [],
            'reason': 'Node not found'
        }

    # Get hop limit from source node
    hop_limit = simulator.nodes[from_node].hop_limit

    # Use flood simulation to see how message propagates
    flood_result = simulate_broadcast(from_node, hop_limit)

    # Check if destination was reached
    destination_reached = False
    for node_info in flood_result.get('nodes', []):
        if node_info['id'] == to_node and node_info['status'] == 'received':
            destination_reached = True
            break

    # Build path and hop details
    path = []
    hops = []
    total_latency = 0

    if destination_reached:
        path = reconstruct_path(flood_result, from_node, to_node)

        for i, node_id in enumerate(path):
            node = simulator.nodes[node_id]
            hop_info = {
                'node': node_id,
                'name': node.name,
                'hop': i,
                'role': node.role
            }

            if i > 0:
                prev_node = simulator.nodes[path[i - 1]]
                link = simulator.calculate_link_quality(prev_node, node)
                hop_info['rssi'] = float(link['rssi'])
                hop_info['snr'] = float(link['snr'])
                hop_info['distance'] = float(link['distance'])
                hop_info['quality'] = int(link['signalQuality'])
                latency = link['distance'] / 300000 * 1000 + 50
                hop_info['latency'] = float(round(latency, 1))
                total_latency += latency
            else:
                hop_info['rssi'] = 0
                hop_info['snr'] = 0
                hop_info['distance'] = 0
                hop_info['latency'] = 0

            hops.append(hop_info)

    return {
        'reachable': destination_reached,
        'source': from_node,
        'destination': to_node,
        'path': path,
        'hops': hops,
        'totalHops': len(path) - 1 if path else 0,
        'totalLatency': float(round(total_latency, 1)),
        'propagation': flood_result.get('propagation', []),
        'floodResult': flood_result,
        'hopLimit': hop_limit,
        'reason': None if destination_reached else f'Message died after {hop_limit} hops - destination not reached'
    }


def calculate_route(from_node: int, to_node: int, hop_limit: int = None) -> List[int]:
    """Calculate a route between two nodes (BFS) respecting hop limit.

    If hop_limit is provided, only returns routes within that many hops.
    If hop_limit is None, uses the source node's configured hop limit.
    """
    if from_node not in simulator.nodes or to_node not in simulator.nodes:
        return []

    # Get hop limit from source node if not specified
    if hop_limit is None:
        hop_limit = simulator.nodes[from_node].hop_limit

    logger.debug(f"[ROUTE] Calculating route from {from_node} to {to_node} with hop_limit={hop_limit}")

    # Build adjacency list
    adj = {n: [] for n in simulator.nodes}
    for n1_id, n1 in simulator.nodes.items():
        for n2_id, n2 in simulator.nodes.items():
            if n1_id != n2_id:
                quality = simulator.calculate_link_quality(n1, n2)
                if quality['canReceive']:
                    adj[n1_id].append(n2_id)

    # BFS with hop limit
    visited = {from_node}
    queue_list = [[from_node]]

    while queue_list:
        path = queue_list.pop(0)
        node = path[-1]

        # Check if we've exceeded hop limit (path length - 1 = number of hops)
        current_hops = len(path) - 1
        if current_hops > hop_limit:
            continue  # Skip paths that exceed hop limit

        if node == to_node:
            logger.debug(f"[ROUTE] Found route: {path} ({len(path)-1} hops)")
            return path

        # Only explore further if we haven't reached the hop limit
        if current_hops < hop_limit:
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue_list.append(path + [neighbor])

    logger.debug(f"[ROUTE] No route found within {hop_limit} hops")
    return []  # No route found within hop limit


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Meshtastic Web Simulator                           ║
║                                                              ║
║  Open your browser to: http://localhost:{port}                ║
║                                                              ║
║  Features:                                                   ║
║  - Visual node placement on a map                            ║
║  - Real-time link quality calculation                        ║
║  - Network topology visualization                            ║
║  - Export/Import node configurations                         ║
║  - Simulated message routing                                 ║
╚══════════════════════════════════════════════════════════════╝
    """)

    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)
