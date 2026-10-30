import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

from verify_korea_boundary import PackedBoundary, require


ROOT_SHA256 = '8cb20a5c0da9c3d6225efb0323e24a4f992eb3029a772da3a71a55d3f503bf17'
SOURCE_SHA256 = '5060f88241e2d80b87f052b4b83296ef59cd7f4ee066a9330bc2ea580bc55428'
ATLAS_SHA256 = {
    4: 'f3005c68a3ce34d2816acfd4e09e5f6c3fd29e925d2ba62f2d152327d5df4f63',
    8: 'a55dcf2e37a4500d6c30de3f5dad42131758415bcfafac79e88d76fb88b0e84d',
}


def original_mask(boundary):
    fragments = [[] for _ in range(boundary.ring_count)]
    owners = {}
    for tile, (state, _, _) in enumerate(boundary.tiles):
        if state != 2:
            continue
        for page in boundary.tile_pages(tile):
            if page[0] == 1:
                _, ring, polygon, first, hole, coordinates = page
                require(ring not in owners or owners[ring] == (polygon, hole),
                        'Conflicting source ring ownership')
                owners[ring] = polygon, hole
                fragments[ring].append((first, coordinates))
    groups = [[] for _ in range(boundary.polygon_count)]
    rings = []
    for ring, pieces in enumerate(fragments):
        require(bool(pieces), 'Missing original source ring')
        count = max(first + len(coordinates) for first, coordinates in pieces)
        restored = np.empty((count, 2), dtype='<f8')
        covered = np.zeros(count, dtype=bool)
        for first, coordinates in pieces:
            end = first + len(coordinates)
            existing = covered[first:end]
            require(np.array_equal(restored[first:end][existing].view('u8'),
                                   coordinates[existing].view('u8')),
                    'Conflicting original source coordinate bits')
            restored[first:end] = coordinates
            covered[first:end] = True
        require(np.all(covered) and np.array_equal(restored[0].view('u8'), restored[-1].view('u8')),
                'Incomplete or open original source ring')
        owner, hole = owners[ring]
        groups[owner].append((hole, restored))
        rings.append(restored)
    require(sum(map(len, rings)) == boundary.vertices, 'Original source vertex count mismatch')
    polygons = []
    for group in groups:
        exteriors = [coordinates for hole, coordinates in group if hole == 0]
        interiors = [coordinates for hole, coordinates in group if hole == 1]
        require(len(exteriors) == 1, 'Original exterior ring count mismatch')
        polygons.append(Polygon(exteriors[0], interiors))
    mask = MultiPolygon(polygons)
    require(mask.is_valid and not mask.is_empty, 'Invalid original source topology')
    require(np.array_equal(shapely.get_coordinates(mask).view('u8'), np.concatenate(rings).view('u8')),
            'GEOS changed original source coordinate bits')
    return mask


def atlas_bytes(boundary, mask, subdivision):
    mixed_tiles = [tile for tile, fields in enumerate(boundary.tiles) if fields[0] == 2]
    per_tile = subdivision * subdivision
    states = np.empty(len(mixed_tiles) * per_tile, dtype=np.uint8)
    columns = np.tile(np.arange(subdivision), subdivision)
    rows = np.repeat(np.arange(subdivision), subdivision)
    step = 1 / (boundary.scale * subdivision)
    shapely.prepare(mask)
    for ordinal, tile in enumerate(mixed_tiles):
        row, column = divmod(tile, boundary.columns)
        west = boundary.grid_west + column / boundary.scale + columns * step
        south = boundary.grid_south + row / boundary.scale + rows * step
        cells = shapely.box(west, south, west + step, south + step)
        inside = shapely.covers(mask, cells)
        outside = shapely.disjoint(mask, cells)
        require(not np.any(inside & outside), 'Inconsistent whole-cell classification')
        states[ordinal * per_tile:(ordinal + 1) * per_tile] = np.where(inside, 1, np.where(outside, 0, 2))
    packed = np.zeros(len(states) // 4, dtype=np.uint8)
    for shift in range(4):
        packed |= states[shift::4] << (shift * 2)
    header = struct.pack('<8s4I32s32s2I', b'SRGKA001', 96, subdivision, len(mixed_tiles),
                         len(boundary.tiles), bytes.fromhex(ROOT_SHA256), boundary.digest, len(packed), 0)
    return header + packed.tobytes(), np.bincount(states, minlength=3).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('boundary', type=Path)
    parser.add_argument('--subdivision', type=int, choices=sorted(ATLAS_SHA256), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = (args.boundary / 'index.bin').read_bytes()
    require(hashlib.sha256(root).hexdigest() == ROOT_SHA256, 'Unexpected original root SHA-256')
    boundary = PackedBoundary(args.boundary)
    require(boundary.digest.hex() == SOURCE_SHA256, 'Unexpected original source SHA-256')
    blob, counts = atlas_bytes(boundary, original_mask(boundary), args.subdivision)
    digest = hashlib.sha256(blob).hexdigest()
    require(digest == ATLAS_SHA256[args.subdivision], 'Atlas differs from the independently certified geometry')
    args.output.write_bytes(blob)
    print(json.dumps({'file': str(args.output), 'subdivision': args.subdivision, 'bytes': len(blob),
                      'sha256': digest, 'states_outside_inside_mixed': counts,
                      'root_sha256': ROOT_SHA256, 'source_sha256': SOURCE_SHA256,
                      'numpy': np.__version__, 'shapely': shapely.__version__,
                      'geos': shapely.geos_version_string}, indent=2))


if __name__ == '__main__':
    main()
