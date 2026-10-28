import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np
import pyproj
import shapefile
import shapely
from shapely.geometry import box, shape


def shard_output(source, manifest):
    blob = source.read_bytes()
    magic, blocks, triangles, record, _, weight = struct.unpack_from('<8s4IQ', blob)
    directory = source.with_suffix('')
    directory.mkdir(exist_ok=True)
    parts = []
    payload = bytearray()
    index = []
    for block in range(blocks):
        end, offset, count, size = struct.unpack_from('<QQII', blob, 32 + block * 24)
        if len(payload) + size > 524288:
            parts.append(bytes(payload))
            payload = bytearray()
        locator = (len(parts) << 32) | len(payload)
        payload.extend(blob[offset:offset + size])
        index.append((end, locator, count, size))
    if payload:
        parts.append(bytes(payload))
    header = struct.pack('<8s4IQ', magic, blocks, triangles, record, len(parts), weight)
    (directory / 'index.bin').write_bytes(header + b''.join(struct.pack('<QQII', *entry) for entry in index))
    for number, data in enumerate(parts):
        (directory / f'part-{number:03}.bin').write_bytes(data)
    module = Path('src/korea_boundary/data.rs')
    module.parent.mkdir(parents=True, exist_ok=True)
    import os
    prefix = Path(os.path.relpath(directory, module.parent)).as_posix()
    module.write_text('pub(super) static INDEX: &[u8] = include_bytes!("' + prefix + '/index.bin");\n'
        + 'pub(super) static PARTS: [&[u8]; ' + str(len(parts)) + '] = [\n'
        + ''.join('    include_bytes!("' + prefix + f'/part-{number:03}.bin"),\n' for number in range(len(parts))) + '];\n')
    manifest.pop("sha256", None)
    manifest.update(parts=len(parts), max_compressed_part_bytes=max(map(len, parts)), hot_area_threshold_m2=10000,
        triangle_order='Stable geographical order within area >= 10000 m2 first, then smaller faces',
        files=[{'file':p.name, 'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(directory.glob('*.bin'))])
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    source.unlink()
    source.with_suffix('.json').unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/korea-boundary-v1.bin'))
    args = parser.parse_args()
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_sha != '5060f88241e2d80b87f052b4b83296ef59cd7f4ee066a9330bc2ea580bc55428':
        raise ValueError('Unexpected corrected boundary source')
    reader = shapefile.Reader(str(args.source))
    source = shape(reader.shape(0).__geo_interface__)
    crs = pyproj.CRS.from_wkt(args.source.with_suffix('.prj').read_text())
    transformer = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    mask = shapely.transform(source, transformer.transform, interleaved=False)
    if not mask.is_valid:
        raise ValueError(shapely.is_valid_reason(mask))
    geod = pyproj.Geod(ellps='GRS80')
    components = np.asarray(list(mask.geoms), dtype=object)
    tree = shapely.STRtree(components)
    west, south, east, north = mask.bounds
    records = 0
    total_weight = 0
    total_planar_area = 0.0
    max_coverage_error = 0.0
    tiles = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as raw:
        for iy in range(math.floor(south * 4), math.ceil(north * 4)):
            for ix in range(math.floor(west * 4), math.ceil(east * 4)):
                window = box(ix / 4, iy / 4, (ix + 1) / 4, (iy + 1) / 4)
                indices = tree.query(window, predicate='intersects')
                if len(indices) == 0:
                    continue
                clipped = shapely.union_all(shapely.intersection(components[indices], window))
                if clipped.area == 0:
                    continue
                triangles = shapely.constrained_delaunay_triangles(clipped)
                faces = np.asarray(list(triangles.geoms), dtype=object)
                if len(faces) == 0 or not shapely.coverage_is_valid(faces):
                    raise ValueError('Invalid constrained triangulation')
                face_area = shapely.area(faces)
                delta = abs(float(face_area.sum()) - clipped.area)
                if delta > max(1e-18, clipped.area * 2e-12):
                    raise ValueError('Triangulation does not cover source tile')
                max_coverage_error = max(max_coverage_error, delta)
                total_planar_area += clipped.area
                coords = shapely.get_coordinates(triangles).reshape((-1, 4, 2))[:, :3, :]
                center = coords.mean(axis=1)
                order = np.lexsort((center[:, 1], center[:, 0]))
                for points in coords[order]:
                    area, _ = geod.polygon_area_perimeter(points[:, 0], points[:, 1])
                    weight = max(1, round(abs(area) * 1_000_000))
                    total_weight += weight
                    raw.write(struct.pack('<Q6d', weight, *points[:, ::-1].ravel()))
                    records += 1
                tiles += 1
                print(json.dumps({'tile': tiles, 'triangles': records}), flush=True)
        if abs(total_planar_area - mask.area) > mask.area * 2e-11:
            raise ValueError('Tiling lost source area')
        block_count = (records + 255) // 256
        prefix_size = 32 + block_count * 24
        running = 0
        index = []
        def ordered_records():
            for hot in [True, False]:
                raw.seek(0)
                while record := raw.read(56):
                    weight = struct.unpack_from('<Q', record)[0]
                    if (weight >= 10_000_000_000) == hot:
                        yield record
        ordered = iter(ordered_records())
        with args.output.open('w+b') as output:
            output.seek(prefix_size)
            previous_end = (1 << 64) - 1
            for block in range(block_count):
                count = min(256, records - block * 256)
                vertices = []
                vertex_index = {}
                faces = []
                for _ in range(count):
                    record = next(ordered)
                    weight = struct.unpack_from('<Q', record)[0]
                    running += weight
                    end = (running << 64) // total_weight - 1
                    coordinates = struct.unpack_from('<6d', record, 8)
                    indices = []
                    for i in range(0, 6, 2):
                        vertex = coordinates[i:i + 2]
                        if vertex not in vertex_index:
                            vertex_index[vertex] = len(vertices)
                            vertices.append(vertex)
                        indices.append(vertex_index[vertex])
                    width = (end - previous_end) & ((1 << 64) - 1)
                    previous_end = end
                    faces.append((width, *indices))
                vertex_bytes = np.asarray(vertices, dtype='<f8').view('u1').reshape(len(vertices) * 2, 8).T.tobytes()
                face_bytes = b''.join(struct.pack('<Q3H', *face) for face in faces)
                face_bytes = np.frombuffer(face_bytes, dtype=np.uint8).reshape(count, 14).T.tobytes()
                encoded = struct.pack('<HH', len(vertices), count) + vertex_bytes + face_bytes
                compressed = zlib.compress(encoded, 9)
                offset = output.tell()
                output.write(compressed)
                index.append((end, offset, count, len(compressed)))
            output.seek(0)
            output.write(struct.pack('<8s4IQ', b'SRGKR001', block_count, records, 14, 256, total_weight))
            for row in index:
                output.write(struct.pack('<QQII', *row))
    result = {'version': 1, 'codec': 'zlib: byte-shuffled f64 vertices, width-u64 + three-u16 vertex indices', 'source_sha256': source_sha, 'source_crs': 'EPSG:5179', 'triangulation_crs': 'EPSG:4326', 'selection_weights': 'GRS80 geodesic triangle area rounded to square millimetre units; minimum 1', 'triangles': records, 'blocks': block_count, 'tile_count': tiles, 'vertices_simplified': False, 'polygon_components': len(components), 'planar_coverage_error_deg2': abs(total_planar_area - mask.area), 'max_tile_coverage_error_deg2': max_coverage_error, 'total_weight_units': total_weight, 'bytes': args.output.stat().st_size, 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'versions': {'shapely': shapely.__version__, 'geos': shapely.geos_version_string, 'pyproj': pyproj.__version__, 'numpy': np.__version__}}
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    shard_output(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
