# Meshtastic Network Simulator - Web Interface

> **ALPHA SOFTWARE** - This project is under rapid development. Features may break, change, or behave unexpectedly. New features are being added frequently. Use at your own risk and expect updates!

A web-based visual simulator for [Meshtastic](https://meshtastic.org/) mesh networks, providing an easy-to-use graphical interface for network planning and message propagation simulation.

Built as a wrapper around [Meshtasticator](https://github.com/meshtastic/Meshtasticator), this tool makes mesh network simulation accessible to everyone - no Linux expertise or CLI knowledge required.

![Status](https://img.shields.io/badge/status-alpha-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

## Features
/Recording%202025-12-14%20142541.mp4
### Visual Network Design
- **Interactive Map Canvas** - OpenStreetMap background with drag-to-position nodes
- **Click-to-Add Nodes** - Double-click anywhere to place nodes
- **Real-time Link Quality** - See RSSI, SNR, and signal quality between nodes
- **Network Topology View** - Visualize all connections and coverage

### Message Simulation with Animation
- **Broadcast Simulation** - Watch messages propagate hop-by-hop through the mesh
- **Direct Message Routing** - See how DMs find their path to the destination
- **Traceroute Visualization** - Discover network paths with detailed hop information
- **Animated Packets** - Visual packets travel between nodes showing RSSI values

### Node Configuration
- **Multiple Roles** - Client, Client Mute, Router, Repeater
- **Antenna Settings** - Height and gain configuration
- **Hop Limit Control** - Set message hop limits per node
- **Drag to Reposition** - Move nodes and see coverage changes

### Path Loss Models
Choose from 7 propagation models for accurate simulation:
| Model | Best For |
|-------|----------|
| Log-distance | Generic calculations |
| Okumura-Hata (small cities) | Urban with low buildings |
| Okumura-Hata (metropolitan) | Dense urban areas |
| Okumura-Hata (suburban) | Residential areas |
| Okumura-Hata (rural) | Open countryside |
| 3GPP (suburban macro) | Mixed suburban |
| 3GPP (metro macro) | City centers |

### Import/Export
- **YAML Configuration** - Save and load node setups
- **Meshtasticator Compatible** - Export configs for use with CLI tools

## Quick Start

### Option 1: Run Locally (Linux)

```bash
cd web-simulator
./run.sh
```

Then open http://localhost:4000 in your browser.

### Option 2: Docker

```bash
cd web-simulator
docker-compose up -d
```

Then open http://localhost:4000 in your browser.

### Option 3: Manual Installation

```bash
# Clone this repo
git clone https://github.com/IceNet-01/meshtastic-simulator.git
cd meshtastic-simulator

# Clone upstream Meshtasticator
git clone https://github.com/meshtastic/Meshtasticator.git upstream-meshtasticator

# Set up web simulator
cd web-simulator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python app.py
```

## Usage

### Adding Nodes
1. **Double-click on map** - Add node at that location
2. **Use the form** - Enter exact coordinates in the left panel
3. **Configure** - Set role, hop limit, antenna gain, and height

### Running Simulations
1. Add at least 2 nodes to the network
2. Select source/destination in the Commands panel
3. Click **Send Broadcast**, **Send DM**, or **Run Traceroute**
4. Watch the animated visualization on the map
5. Check the Message Log for detailed results

### Understanding Results
- **Cyan packets** - Broadcast messages
- **Green packets** - Direct messages
- **Yellow packets** - Traceroute probes
- **Green rings** - Nodes that received the message
- **Red X** - Nodes that couldn't receive
- **H1, H2, etc.** - Hop count labels

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/config` | Get simulator configuration |
| POST | `/api/config` | Update configuration |
| GET | `/api/nodes` | List all nodes |
| POST | `/api/nodes` | Add a new node |
| PUT | `/api/nodes/<id>` | Update a node |
| DELETE | `/api/nodes/<id>` | Delete a node |
| POST | `/api/nodes/clear` | Remove all nodes |
| GET | `/api/topology` | Get network topology with links |
| GET | `/api/link/<id1>/<id2>` | Get link quality between two nodes |
| GET | `/api/export/yaml` | Export node config as YAML |
| POST | `/api/import/yaml` | Import node config from YAML |

### WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client → Server | Connection established |
| `node_added` | Server → Client | New node added |
| `node_updated` | Server → Client | Node configuration changed |
| `node_removed` | Server → Client | Node deleted |
| `send_command` | Client → Server | Execute simulator command |
| `command_response` | Server → Client | Command result with simulation data |

## Keyboard & Mouse Controls

- **Double-click canvas** - Add node at location
- **Click node** - Select node
- **Shift+Click node** - Multi-select nodes
- **Drag node** - Reposition node
- **Drag empty space** - Pan the map
- **Scroll wheel** - Zoom in/out
- **Fit button** - Auto-zoom to show all nodes

## Roadmap

Features coming soon:
- [ ] Real GPS coordinate input
- [ ] Terrain/elevation data integration
- [ ] Channel configuration
- [ ] Message encryption simulation
- [ ] Network statistics dashboard
- [ ] Multiple simultaneous simulations
- [ ] Save/load simulation scenarios
- [ ] Collaborative editing

## Known Issues

- Map tiles may load slowly on first view
- Animation performance may vary on large networks (20+ nodes)
- Some path loss models may give unexpected results at extreme distances

## Contributing

Contributions welcome! This project aims to make Meshtastic network planning accessible to everyone.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - Same as Meshtasticator

## Credits

- [Meshtasticator](https://github.com/meshtastic/Meshtasticator) - The underlying simulation engine
- [Meshtastic](https://meshtastic.org/) - The mesh networking project
- [OpenStreetMap](https://www.openstreetmap.org/) - Map tiles

## Disclaimer

This is an independent project and is not officially affiliated with or endorsed by the Meshtastic project. Use simulation results as guidance only - real-world performance will vary based on terrain, obstacles, interference, and other factors.
