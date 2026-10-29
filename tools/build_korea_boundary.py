import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import struct

import numpy as np
import pyproj
import shapefile
import shapely
from shapely.geometry import box, shape

from pack_korea_boundary import (
    BoundaryPacker, MAX_CHAIN_EDGES, MAX_PARITY_EVENTS, write_rust_module,
)
from uniform_math import rectangle_surface_area, surface_area


SOURCE_SHA256 = '5060f88241e2d80b87f052b4b83296ef59cd7f4ee066a9330bc2ea580bc55428'


def load_source(source):
    source = Path(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Unexpected corrected boundary source')
    crs = pyproj.CRS.from_wkt(source.with_suffix('.prj').read_text())
    if crs.to_epsg() != 5179:
        raise ValueError('Unexpected corrected boundary coordinate system')
    reader = shapefile.Reader(str(source))
    if len(reader) != 1:
        raise ValueError('Expected one corrected boundary record')
    original = shape(reader.shape(0).__geo_interface__)
    transformer = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    mask = shapely.transform(original, transformer.transform, interleaved=False)
    if not original.is_valid or not mask.is_valid or mask.geom_type != 'MultiPolygon':
        raise ValueError('Invalid corrected boundary topology')
    return mask


def polygon_rings(mask):
    polygons = [mask] if mask.geom_type == 'Polygon' else list(mask.geoms)
    rings = []
    owners = []
    holes = []
    for owner, polygon in enumerate(polygons):
        for hole, ring in enumerate([polygon.exterior, *polygon.interiors]):
            vertices = np.asarray(ring.coords, dtype='<f8')
            if len(vertices) < 4 or not np.array_equal(vertices[0], vertices[-1]):
                raise ValueError('Invalid source ring')
            rings.append(vertices)
            owners.append(owner)
            holes.append(int(hole != 0))
    return polygons, rings, owners, holes


def parity_events(start, end, south, north):
    lower = np.minimum(start[:, 1], end[:, 1])
    upper = np.maximum(start[:, 1], end[:, 1])
    base = int(np.count_nonzero((lower <= south) & (south < upper)) % 2)
    endpoints = np.concatenate([lower, upper])
    endpoints = endpoints[(endpoints > south) & (endpoints <= north)]
    values, counts = np.unique(endpoints, return_counts=True)
    return base, values[counts % 2 != 0]


def append_tile(packer, tile, selected, remote_start, remote_end,
                south, north, ring_ids, ring_offsets, rings, owners, holes):
    base, events = parity_events(remote_start, remote_end, south, north)
    for offset in range(0, max(1, len(events)), MAX_PARITY_EVENTS):
        part = np.asarray(events[offset:offset + MAX_PARITY_EVENTS], dtype='<f8')
        packer.append(tile, 0, len(part), base if offset == 0 else 0, 0, part.tobytes())
    if len(selected) == 0:
        raise ValueError('Mixed tile has no source edges')
    selected_rings = ring_ids[selected]
    breaks = np.flatnonzero((np.diff(selected) != 1) | (np.diff(selected_rings) != 0)) + 1
    limits = np.r_[0, breaks, len(selected)]
    for begin, end in zip(limits[:-1], limits[1:], strict=True):
        ring_id = int(selected_rings[begin])
        first = int(selected[begin] - ring_offsets[ring_id])
        count = int(end - begin)
        for offset in range(0, count, MAX_CHAIN_EDGES):
            length = min(MAX_CHAIN_EDGES, count - offset)
            vertices = rings[ring_id][first + offset:first + offset + length + 1]
            coordinates = vertices.view('u1').reshape(-1, 8).T.tobytes()
            payload = struct.pack('<2I', first + offset, holes[ring_id]) + coordinates
            packer.append(tile, 1, length, ring_id, owners[ring_id], payload)


def build_mask(mask, directory, source_digest, scale=4, provenance=None):
    if not mask.is_valid or mask.is_empty or scale not in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        raise ValueError('Invalid polygon mask or tile scale')
    polygons, rings, owners, holes = polygon_rings(mask)
    bounds = tuple(map(float, mask.bounds))
    west, south, east, north = bounds
    if not (124 <= west < east <= 132 and 32 <= south < north <= 40):
        raise ValueError('Boundary exceeds deterministic numerical domain')
    origin_x, origin_y = math.floor(west * scale), math.floor(south * scale)
    columns = math.ceil(east * scale) - origin_x
    rows = math.ceil(north * scale) - origin_y
    if columns * rows > 262144:
        raise ValueError('Boundary tile count limit exceeded')
    grid_west, grid_south = origin_x / scale, origin_y / scale
    source_vertices = sum(map(len, rings))
    counts = np.asarray([len(ring) - 1 for ring in rings], dtype=np.int64)
    ring_offsets = np.r_[0, np.cumsum(counts)[:-1]]
    ring_ids = np.repeat(np.arange(len(rings), dtype=np.int32), counts)
    start = np.concatenate([ring[:-1] for ring in rings])
    end = np.concatenate([ring[1:] for ring in rings])
    lower = np.minimum(start, end)
    upper = np.maximum(start, end)
    packer = BoundaryPacker(directory, source_digest)
    tiles = []
    tile_states = [0, 0, 0]
    max_tile_pages = 0
    max_tile_edges = 0
    stored_edges = 0
    shapely.prepare(mask)
    for row in range(rows):
        tile_south = (origin_y + row) / scale
        tile_north = (origin_y + row + 1) / scale
        row_indices = np.flatnonzero((lower[:, 1] <= tile_north) & (upper[:, 1] >= tile_south))
        row_lower = lower[row_indices]
        row_upper = upper[row_indices]
        for column in range(columns):
            tile_west = (origin_x + column) / scale
            tile_east = (origin_x + column + 1) / scale
            local = (row_lower[:, 0] <= tile_east) & (row_upper[:, 0] >= tile_west)
            selected = row_indices[local]
            tile = row * columns + column
            first_page = packer.page_count
            if len(selected):
                remote = row_indices[row_lower[:, 0] > tile_east]
                append_tile(
                    packer, tile, selected, start[remote], end[remote],
                    tile_south, tile_north, ring_ids, ring_offsets, rings, owners, holes,
                )
                state = 2
                max_tile_edges = max(max_tile_edges, len(selected))
                stored_edges += len(selected)
            else:
                cell = box(tile_west, tile_south, tile_east, tile_north)
                if mask.covers(cell):
                    state = 1
                elif mask.disjoint(cell):
                    state = 0
                else:
                    raise ValueError('Tile classification missed a source boundary')
            page_count = packer.page_count - first_page
            max_tile_pages = max(max_tile_pages, page_count)
            tiles.append((state, first_page, page_count))
            tile_states[state] += 1
        print(json.dumps({'row': row + 1, 'rows': rows, 'pages': packer.page_count}), flush=True)
    root = packer.finish(
        tiles, columns, rows, scale, grid_west, grid_south, bounds,
        len(polygons), len(rings), source_vertices,
    )
    files = [{'file': path.name, 'bytes': path.stat().st_size,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(Path(directory).glob('*.bin'))]
    area = surface_area(mask)
    rectangle_area = rectangle_surface_area(bounds)
    manifest = {
        'version': 4,
        'source_sha256': source_digest,
        'source_crs': 'EPSG:5179',
        'mask_crs': 'EPSG:4326',
        'polygon_components': len(polygons),
        'rings': len(rings),
        'source_vertices': source_vertices,
        'bounds': dict(zip(['west', 'south', 'east', 'north'], bounds, strict=True)),
        'bounds_expansion_ulps': 0,
        'tile_scale': scale,
        'tile_degrees': 1 / scale,
        'grid_origin': [grid_west, grid_south],
        'columns': columns,
        'rows': rows,
        'tile_count': len(tiles),
        'tile_states': dict(zip(['OUTSIDE', 'INSIDE', 'MIXED'], tile_states, strict=True)),
        'parts': len(packer.parts),
        'pages': packer.page_count,
        'max_tile_pages': max_tile_pages,
        'max_tile_edges': max_tile_edges,
        'stored_edges_with_tile_duplicates': stored_edges,
        'raw_page_bytes': packer.raw_bytes,
        'max_raw_page_bytes': packer.max_page_bytes,
        'max_compressed_part_bytes': max(part[2] for part in packer.parts),
        'root_index_bytes': len(root),
        'root_index_sha256': hashlib.sha256(root).hexdigest(),
        'bytes': sum(entry['bytes'] for entry in files),
        'files': files,
        'vertices_simplified': False,
        'vertices_rounded': False,
        'geometry_clipped': False,
        'codec': 'zlib; byte-shuffled original f64 ring chains; exact right-ray parity events',
        'target_measure': 'GRS80 surface area on the unchanged piecewise-linear EPSG:4326 mask',
        'uniformity': 'Equal probability on accepted centers of a 2^32 by 2^32 equal-area grid; whole independent u64 rejection',
        'surface_area_m2': area,
        'bounding_rectangle_surface_area_m2': rectangle_area,
        'expected_acceptance_probability': area / rectangle_area,
        'expected_supplemental_reads': rectangle_area / area - 1,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'shapely': shapely.__version__, 'geos': shapely.geos_version_string,
                     'pyproj': pyproj.__version__, 'proj': pyproj.proj_version_str,
                     'pyshp': shapefile.__version__},
    }
    if provenance is not None:
        manifest['provenance'] = provenance
    (Path(directory) / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest, root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/korea-boundary-v4'))
    parser.add_argument('--tile-scale', type=int, default=32)
    parser.add_argument('--rust-module', type=Path, default=Path('src/korea_boundary/data.rs'))
    parser.add_argument('--boundary-validation', type=Path)
    args = parser.parse_args()
    mask = load_source(args.source)
    polygons, rings, _, _ = polygon_rings(mask)
    if (len(polygons), len(rings), sum(map(len, rings))) != (13701, 13713, 5205473):
        raise ValueError('Corrected source geometry counts changed')
    validation_path = args.boundary_validation or args.source.parent / 'validation.json'
    validation = json.loads(validation_path.read_text())
    probes = validation['regression_probes']
    if len(probes) != 15:
        raise ValueError('Expected all 15 original geographic probes')
    for name, probe in probes.items():
        actual = mask.covers(shapely.Point(probe['lon'], probe['lat']))
        if actual != probe['expected_inside']:
            raise ValueError(f'Original geographic probe failed: {name}')
    provenance = {
        'reproduction_command': [
            'python', 'tools/build_korea_boundary.py', 'corrected/' + args.source.name,
            '--output', 'rebuilt/korea-boundary-v4', '--tile-scale', str(args.tile_scale),
            '--rust-module', 'rebuilt/data.rs', '--boundary-validation',
            'data/korea-boundary-v4/boundary-validation.json',
        ],
        'inputs': [{'file': path.name, 'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in sorted(args.source.parent.glob(args.source.stem + '.*'))],
        'reference_validation': {
            'file': 'boundary-validation.json', 'bytes': validation_path.stat().st_size,
            'sha256': hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        },
    }
    manifest, root = build_mask(mask, args.output, SOURCE_SHA256, args.tile_scale, provenance)
    from verify_korea_boundary import validate_reconstruction
    result = validate_reconstruction(args.output, mask)
    result['original_geographic_probes_passed'] = len(probes)
    (args.output / 'geometry-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copyfile(validation_path, args.output / 'boundary-validation.json')
    write_rust_module(args.output, args.rust_module, root)
    print(json.dumps({key: manifest[key] for key in [
        'version', 'polygon_components', 'rings', 'source_vertices', 'tile_count',
        'tile_states', 'parts', 'pages', 'bytes', 'root_index_sha256',
    ]}), flush=True)


if __name__ == '__main__':
    main()
