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
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any

import yaml
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

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

    def to_dict(self) -> dict:
        return {
            'id': self.node_id,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'role': self.role,
            'hopLimit': self.hop_limit,
            'antennaGain': self.antenna_gain,
            'name': self.name,
            'neighborInfo': self.neighbor_info
        }

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
        path_loss = phy.estimate_path_loss(self.config, dist, self.config.FREQ, node1.z, node2.z)
        rssi = self.config.PTX + node1.antenna_gain - path_loss
        snr = rssi - self.config.NOISE_LEVEL
        can_receive = bool(rssi >= self.config.SENSMODEM[self.config.MODEM])

        return {
            'distance': float(round(dist, 1)),
            'pathLoss': float(round(path_loss, 2)),
            'rssi': float(round(rssi, 2)),
            'snr': float(round(snr, 2)),
            'canReceive': can_receive,
            'signalQuality': int(self._rssi_to_quality(rssi)) if can_receive else 0
        }

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
    """Update simulator configuration."""
    data = request.json
    if 'model' in data:
        simulator.config.MODEL = int(data['model'])
    if 'xsize' in data:
        simulator.config.XSIZE = float(data['xsize'])
    if 'ysize' in data:
        simulator.config.YSIZE = float(data['ysize'])
    if 'hopLimit' in data:
        simulator.config.hopLimit = int(data['hopLimit'])
    if 'defaultHeight' in data:
        simulator.config.HM = float(data['defaultHeight'])
    if 'defaultGain' in data:
        simulator.config.GL = float(data['defaultGain'])
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


@app.route('/api/export/yaml')
def export_yaml():
    """Export node configuration as YAML."""
    yaml_content = simulator.get_node_config_yaml()
    return yaml_content, 200, {'Content-Type': 'text/yaml'}


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
            role = 'CLIENT'
            if node_config.get('isRouter'):
                role = 'ROUTER'
            elif node_config.get('isRepeater'):
                role = 'REPEATER'
            elif node_config.get('isClientMute'):
                role = 'CLIENT_MUTE'

            simulator.add_node(
                x=node_config['x'],
                y=node_config['y'],
                z=node_config.get('z', 1.0),
                role=role,
                hop_limit=node_config.get('hopLimit', 3),
                antenna_gain=node_config.get('antennaGain', 0),
                neighbor_info=node_config.get('neighborInfo', False)
            )

        socketio.emit('config_imported', {'nodeCount': len(simulator.nodes)})
        return jsonify({'status': 'ok', 'nodeCount': len(simulator.nodes)})

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

        # Simulate broadcast propagation
        result = simulate_broadcast(from_node, hop_limit)

        emit('command_response', {
            'command': 'broadcast',
            'status': 'success' if result['totalReceived'] > 0 else 'partial',
            'message': f'Broadcast from Node {from_node}: {result["totalReceived"]}/{result["totalNodes"]-1} nodes received',
            'simulation': result
        }, broadcast=True)

    elif command == 'dm':
        from_node = args.get('from')
        to_node = args.get('to')
        text = args.get('text', 'Test message')

        # Simulate direct message routing
        result = simulate_direct_message(from_node, to_node)

        status = 'success' if result['delivered'] else 'failed'
        msg = f'DM {from_node} → {to_node}: {"Delivered" if result["delivered"] else "Failed"}'
        if result['delivered']:
            msg += f' via {len(result["path"])-1} hop(s)'

        emit('command_response', {
            'command': 'dm',
            'status': status,
            'message': msg,
            'simulation': result
        }, broadcast=True)

    elif command == 'traceroute':
        from_node = args.get('from')
        to_node = args.get('to')

        # Simulate traceroute with detailed hop info
        result = simulate_traceroute(from_node, to_node)

        if result['reachable']:
            hop_info = ' → '.join([f"{h['node']}({h['rssi']:.0f}dBm)" for h in result['hops']])
            msg = f'Traceroute {from_node} → {to_node}: {hop_info}'
        else:
            msg = f'Traceroute {from_node} → {to_node}: No route found'

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
    """Simulate a broadcast message propagating through the mesh."""
    if from_node not in simulator.nodes:
        return {'error': 'Source node not found', 'totalReceived': 0, 'totalNodes': 0, 'propagation': []}

    source = simulator.nodes[from_node]
    all_nodes = list(simulator.nodes.keys())

    # Track which nodes received the message and when (hop count)
    received = {from_node: {'hop': 0, 'rssi': 0, 'from': None}}
    propagation = []  # List of propagation steps for animation

    # Simulate hop-by-hop propagation
    current_hop = 0
    frontier = [from_node]

    while current_hop < hop_limit and frontier:
        next_frontier = []
        hop_transmissions = []

        for tx_node_id in frontier:
            tx_node = simulator.nodes[tx_node_id]

            # Check which nodes can hear this transmission
            for rx_node_id, rx_node in simulator.nodes.items():
                if rx_node_id == tx_node_id:
                    continue
                if rx_node_id in received:
                    continue  # Already received

                # Check if can receive
                link = simulator.calculate_link_quality(tx_node, rx_node)
                if link['canReceive']:
                    received[rx_node_id] = {
                        'hop': current_hop + 1,
                        'rssi': link['rssi'],
                        'snr': link['snr'],
                        'from': tx_node_id
                    }
                    hop_transmissions.append({
                        'from': tx_node_id,
                        'to': rx_node_id,
                        'rssi': float(link['rssi']),
                        'snr': float(link['snr']),
                        'distance': float(link['distance'])
                    })

                    # Only routers/repeaters rebroadcast, not client_mute
                    role = rx_node.role
                    if role in ['ROUTER', 'REPEATER', 'CLIENT']:
                        next_frontier.append(rx_node_id)

        if hop_transmissions:
            propagation.append({
                'hop': current_hop + 1,
                'transmissions': hop_transmissions
            })

        frontier = next_frontier
        current_hop += 1

    # Build result
    node_results = []
    for node_id in all_nodes:
        if node_id == from_node:
            node_results.append({'id': node_id, 'status': 'source', 'hop': 0})
        elif node_id in received:
            r = received[node_id]
            node_results.append({
                'id': node_id,
                'status': 'received',
                'hop': r['hop'],
                'rssi': float(r['rssi']),
                'snr': float(r['snr']),
                'from': r['from']
            })
        else:
            node_results.append({'id': node_id, 'status': 'unreached', 'hop': -1})

    return {
        'source': from_node,
        'hopLimit': hop_limit,
        'totalNodes': len(all_nodes),
        'totalReceived': len(received) - 1,  # Exclude source
        'nodes': node_results,
        'propagation': propagation
    }


def simulate_direct_message(from_node: int, to_node: int) -> dict:
    """Simulate a direct message with acknowledgment."""
    path = calculate_route(from_node, to_node)

    if not path:
        return {
            'delivered': False,
            'path': [],
            'hops': [],
            'reason': 'No route available'
        }

    # Calculate link quality for each hop
    hops = []
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
        'delivered': True,
        'path': path,
        'hops': hops,
        'totalHops': len(path) - 1
    }


def simulate_traceroute(from_node: int, to_node: int) -> dict:
    """Simulate traceroute with detailed hop information."""
    path = calculate_route(from_node, to_node)

    if not path:
        return {
            'reachable': False,
            'path': [],
            'hops': [],
            'reason': 'Destination unreachable'
        }

    # Build detailed hop information
    hops = []
    total_latency = 0

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
            # Estimate latency based on distance (speed of light + processing)
            latency = link['distance'] / 300000 * 1000 + 50  # ms
            hop_info['latency'] = float(round(latency, 1))
            total_latency += latency
        else:
            hop_info['rssi'] = 0
            hop_info['snr'] = 0
            hop_info['distance'] = 0
            hop_info['latency'] = 0

        hops.append(hop_info)

    return {
        'reachable': True,
        'path': path,
        'hops': hops,
        'totalHops': len(path) - 1,
        'totalLatency': float(round(total_latency, 1))
    }


def calculate_route(from_node: int, to_node: int) -> List[int]:
    """Calculate a simple route between two nodes (BFS)."""
    if from_node not in simulator.nodes or to_node not in simulator.nodes:
        return []

    # Build adjacency list
    adj = {n: [] for n in simulator.nodes}
    for n1_id, n1 in simulator.nodes.items():
        for n2_id, n2 in simulator.nodes.items():
            if n1_id != n2_id:
                quality = simulator.calculate_link_quality(n1, n2)
                if quality['canReceive']:
                    adj[n1_id].append(n2_id)

    # BFS
    visited = {from_node}
    queue_list = [[from_node]]

    while queue_list:
        path = queue_list.pop(0)
        node = path[-1]

        if node == to_node:
            return path

        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue_list.append(path + [neighbor])

    return []  # No route found


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
