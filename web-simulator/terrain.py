"""
Terrain/Elevation module for Meshtastic Web Simulator

Provides:
- Elevation data fetching from Open-Elevation API (SRTM data)
- Line-of-sight calculations between nodes
- Terrain obstruction detection
- Fresnel zone clearance calculations
"""

import math
import requests
from typing import List, Tuple, Optional, Dict
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Open-Elevation API endpoint
ELEVATION_API = "https://api.open-elevation.com/api/v1/lookup"

# Cache for elevation data to minimize API calls
elevation_cache: Dict[Tuple[float, float], float] = {}


def get_elevation(lat: float, lon: float) -> Optional[float]:
    """
    Get elevation for a single point using Open-Elevation API.
    Returns elevation in meters above sea level, or None if unavailable.
    """
    cache_key = (round(lat, 5), round(lon, 5))
    if cache_key in elevation_cache:
        return elevation_cache[cache_key]

    try:
        response = requests.post(
            ELEVATION_API,
            json={"locations": [{"latitude": lat, "longitude": lon}]},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                elevation = data["results"][0].get("elevation")
                if elevation is not None:
                    elevation_cache[cache_key] = float(elevation)
                    return float(elevation)
    except Exception as e:
        logger.warning(f"Failed to get elevation for ({lat}, {lon}): {e}")

    return None


def get_elevations_batch(locations: List[Tuple[float, float]]) -> List[Optional[float]]:
    """
    Get elevations for multiple points in a single API call.
    More efficient for checking paths between nodes.
    """
    # Check cache first
    results = []
    uncached_indices = []
    uncached_locations = []

    for i, (lat, lon) in enumerate(locations):
        cache_key = (round(lat, 5), round(lon, 5))
        if cache_key in elevation_cache:
            results.append(elevation_cache[cache_key])
        else:
            results.append(None)
            uncached_indices.append(i)
            uncached_locations.append({"latitude": lat, "longitude": lon})

    if not uncached_locations:
        return results

    try:
        response = requests.post(
            ELEVATION_API,
            json={"locations": uncached_locations},
            timeout=10  # Reduced timeout to prevent long hangs
        )
        if response.status_code == 200:
            data = response.json()
            for idx, result in zip(uncached_indices, data.get("results", [])):
                elevation = result.get("elevation")
                if elevation is not None:
                    lat, lon = locations[idx]
                    cache_key = (round(lat, 5), round(lon, 5))
                    elevation_cache[cache_key] = float(elevation)
                    results[idx] = float(elevation)
    except Exception as e:
        logger.warning(f"Failed to get batch elevations: {e}")

    return results


def sample_path_points(lat1: float, lon1: float, lat2: float, lon2: float,
                       num_samples: int = 10) -> List[Tuple[float, float]]:
    """
    Generate sample points along a path between two coordinates.
    Uses great circle interpolation for accuracy.
    """
    points = []
    for i in range(num_samples + 1):
        t = i / num_samples
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)
        points.append((lat, lon))
    return points


def calculate_fresnel_radius(distance_m: float, freq_mhz: float,
                             d1_m: float, d2_m: float, zone: int = 1) -> float:
    """
    Calculate the radius of the nth Fresnel zone at a point along a path.

    Args:
        distance_m: Total path distance in meters
        freq_mhz: Frequency in MHz
        d1_m: Distance from transmitter to the point
        d2_m: Distance from the point to receiver (should equal distance_m - d1_m)
        zone: Fresnel zone number (default 1 for first zone)

    Returns:
        Fresnel zone radius in meters
    """
    wavelength = 299792458 / (freq_mhz * 1e6)  # wavelength in meters
    # Fresnel zone radius formula: r_n = sqrt(n * lambda * d1 * d2 / (d1 + d2))
    if d1_m <= 0 or d2_m <= 0:
        return 0
    radius = math.sqrt(zone * wavelength * d1_m * d2_m / (d1_m + d2_m))
    return radius


def check_line_of_sight(lat1: float, lon1: float, height1: float,
                        lat2: float, lon2: float, height2: float,
                        freq_mhz: float = 906.0,
                        num_samples: int = 20) -> Dict:
    """
    Check line-of-sight between two nodes considering terrain elevation.

    Args:
        lat1, lon1: Transmitter coordinates
        height1: Transmitter height above ground (AGL) in meters
        lat2, lon2: Receiver coordinates
        height2: Receiver height above ground (AGL) in meters
        freq_mhz: Frequency in MHz (default 906 for US Meshtastic)
        num_samples: Number of terrain samples to check

    Returns:
        Dict with:
        - has_los: Boolean indicating clear line of sight
        - obstruction_loss: Additional path loss from obstructions (dB)
        - clearance_ratio: Fresnel zone clearance (1.0 = clear, <0 = blocked)
        - terrain_profile: List of elevation points along path
        - worst_clearance: Minimum clearance point details
    """
    # Get terrain elevations along the path
    path_points = sample_path_points(lat1, lon1, lat2, lon2, num_samples)
    elevations = get_elevations_batch(path_points)

    # If we couldn't get elevation data, assume clear LOS
    if all(e is None for e in elevations):
        return {
            "has_los": True,
            "obstruction_loss": 0,
            "clearance_ratio": 1.0,
            "terrain_profile": None,
            "worst_clearance": None,
            "error": "Could not fetch elevation data"
        }

    # Fill in missing elevations with interpolation
    valid_elevations = [(i, e) for i, e in enumerate(elevations) if e is not None]
    if len(valid_elevations) < 2:
        return {
            "has_los": True,
            "obstruction_loss": 0,
            "clearance_ratio": 1.0,
            "terrain_profile": None,
            "worst_clearance": None,
            "error": "Insufficient elevation data"
        }

    # Simple linear interpolation for missing values
    for i in range(len(elevations)):
        if elevations[i] is None:
            # Find nearest valid points
            prev_valid = next((j for j in range(i-1, -1, -1) if elevations[j] is not None), None)
            next_valid = next((j for j in range(i+1, len(elevations)) if elevations[j] is not None), None)

            if prev_valid is not None and next_valid is not None:
                t = (i - prev_valid) / (next_valid - prev_valid)
                elevations[i] = elevations[prev_valid] + t * (elevations[next_valid] - elevations[prev_valid])
            elif prev_valid is not None:
                elevations[i] = elevations[prev_valid]
            elif next_valid is not None:
                elevations[i] = elevations[next_valid]

    # Calculate total distance
    distance_m = haversine_distance(lat1, lon1, lat2, lon2)

    # Handle zero or very small distances (same location or very close nodes)
    if distance_m < 1.0:  # Less than 1 meter
        return {
            "has_los": True,
            "obstruction_loss": 0,
            "clearance_ratio": 1.0,
            "terrain_profile": None,
            "worst_clearance": None,
            "distance_m": distance_m,
            "tx_elevation_asl": (elevations[0] or 0) + height1,
            "rx_elevation_asl": (elevations[-1] or 0) + height2
        }

    # Get endpoint elevations (ground level)
    elev1 = elevations[0] if elevations[0] is not None else 0
    elev2 = elevations[-1] if elevations[-1] is not None else 0

    # Antenna heights above sea level
    tx_height_asl = elev1 + height1
    rx_height_asl = elev2 + height2

    # Check each sample point
    worst_clearance_ratio = float('inf')
    worst_point = None
    terrain_profile = []

    for i, (lat, lon) in enumerate(path_points):
        elev = elevations[i]
        if elev is None:
            continue

        # Calculate distance from transmitter to this point
        d1 = (i / num_samples) * distance_m
        d2 = distance_m - d1

        # Calculate LOS height at this point (linear interpolation)
        los_height = tx_height_asl + (rx_height_asl - tx_height_asl) * (d1 / distance_m)

        # Calculate Fresnel zone radius at this point
        fresnel_r = calculate_fresnel_radius(distance_m, freq_mhz, d1, d2)

        # Calculate clearance (LOS height - terrain elevation - required Fresnel clearance)
        # For 60% Fresnel zone clearance (common standard), use 0.6 * fresnel_r
        clearance = los_height - elev - 0.6 * fresnel_r
        clearance_ratio = clearance / (0.6 * fresnel_r) if fresnel_r > 0 else 1.0

        terrain_profile.append({
            "distance": d1,
            "elevation": elev,
            "los_height": los_height,
            "fresnel_radius": fresnel_r,
            "clearance": clearance,
            "clearance_ratio": clearance_ratio
        })

        if clearance_ratio < worst_clearance_ratio:
            worst_clearance_ratio = clearance_ratio
            worst_point = {
                "distance": d1,
                "elevation": elev,
                "los_height": los_height,
                "clearance": clearance,
                "lat": lat,
                "lon": lon
            }

    # Calculate obstruction loss
    # If clearance_ratio < 0, terrain blocks LOS
    # If 0 < clearance_ratio < 1, partial Fresnel zone obstruction
    # Handle edge cases where worst_clearance_ratio might be inf
    if worst_clearance_ratio == float('inf') or worst_clearance_ratio >= 1.0:
        obstruction_loss = 0
        has_los = True
        worst_clearance_ratio = 1.0 if worst_clearance_ratio == float('inf') else worst_clearance_ratio
    elif worst_clearance_ratio > 0:
        # Partial obstruction - use knife-edge diffraction approximation
        # Loss increases as clearance decreases
        obstruction_loss = 6 * (1 - worst_clearance_ratio)  # 0-6 dB for partial obstruction
        has_los = True
    else:
        # Full obstruction - significant diffraction loss
        # Knife-edge diffraction model (simplified)
        obstruction_depth = min(abs(worst_clearance_ratio), 100)  # Cap depth to avoid extreme values
        obstruction_loss = 6 + 10 * math.log10(1 + obstruction_depth)
        obstruction_loss = min(obstruction_loss, 30)  # Cap at 30 dB
        has_los = False

    # Ensure all values are JSON-safe (no inf/nan)
    def safe_round(val, digits=2, default=0.0):
        if val is None or math.isinf(val) or math.isnan(val):
            return default
        return round(val, digits)

    return {
        "has_los": has_los,
        "obstruction_loss": safe_round(obstruction_loss, 2, 0.0),
        "clearance_ratio": safe_round(worst_clearance_ratio, 3, 1.0),
        "terrain_profile": terrain_profile,
        "worst_clearance": worst_point,
        "distance_m": safe_round(distance_m, 1, 0.0),
        "tx_elevation_asl": safe_round(tx_height_asl, 1, 0.0),
        "rx_elevation_asl": safe_round(rx_height_asl, 1, 0.0)
    }


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points in meters.
    """
    R = 6371000  # Earth's radius in meters

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def meters_to_latlon(x_m: float, y_m: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """
    Convert meters offset from reference point to lat/lon.
    """
    # Approximate conversion (works well for small areas)
    lat_offset = y_m / 111320.0
    lon_offset = x_m / (111320.0 * math.cos(math.radians(ref_lat)))

    return ref_lat + lat_offset, ref_lon + lon_offset


def clear_elevation_cache():
    """Clear the elevation cache."""
    global elevation_cache
    elevation_cache.clear()


# Test function
if __name__ == "__main__":
    # Test with two points in San Francisco (hilly terrain)
    # Twin Peaks
    lat1, lon1 = 37.7544, -122.4477
    # Financial District
    lat2, lon2 = 37.7946, -122.3999

    print(f"Testing LOS between Twin Peaks and Financial District")
    result = check_line_of_sight(lat1, lon1, 2.0, lat2, lon2, 2.0)

    print(f"Has LOS: {result['has_los']}")
    print(f"Obstruction Loss: {result['obstruction_loss']} dB")
    print(f"Clearance Ratio: {result['clearance_ratio']}")
    if result.get('worst_clearance'):
        wc = result['worst_clearance']
        print(f"Worst clearance at {wc['distance']:.0f}m: {wc['clearance']:.1f}m")
