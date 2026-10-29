import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib
import numpy as np
from uniform_math import surface_weights


def pack_flat(source, manifest, directory=None):
    blob = source.read_bytes()
    magic, count, triangles, record, _, total = struct.unpack_from('<8s4IQ', blob)
    if magic != b'SRGKR003':
        raise ValueError('Expected uniform v3 flat data')
    directory = directory or source.with_suffix('')
    directory.mkdir(parents=True, exist_ok=True)
    parts, pages = [], []
    size, first = 16, 0

    def flush():
        nonlocal pages, size, first
        if not pages:
            return
        offset = 16 + len(pages) * 16
        entries, payloads = [], []
        for end, compressed in pages:
            entries.append(struct.pack('<QII', end, offset, len(compressed)))
            payloads.append(compressed)
            offset += len(compressed)
        payload = struct.pack('<8sII', b'SRGKP003', len(pages), first) + b''.join(entries + payloads)
        (directory / f'part-{len(parts):03}.bin').write_bytes(payload)
        parts.append((pages[-1][0], len(payload), len(pages)))
        first += len(pages)
        pages, size = [], 16

    for i in range(count):
        end, offset, _, length = struct.unpack_from('<QQII', blob, 32 + i * 24)
        if size + 16 + length > 524288:
            flush()
        pages.append((end, blob[offset:offset + length]))
        size += 16 + length
    flush()
    root = struct.pack('<8s4IQ', b'SRGKR003', len(parts), triangles, record, 256, total)
    root += b''.join(struct.pack('<QII', *part) for part in parts)
    (directory / 'index.bin').write_bytes(root)
    module = Path('src/korea_boundary/data.rs')
    module.parent.mkdir(parents=True, exist_ok=True)
    prefix = Path(os.path.relpath(directory, module.parent)).as_posix()
    module.write_text('pub(super) static INDEX: &[u8] = include_bytes!("' + prefix + '/index.bin");\n'
        + 'pub(super) static PARTS: [&[u8]; ' + str(len(parts)) + '] = [\n'
        + ''.join('    include_bytes!("' + prefix + f'/part-{i:03}.bin"),\n' for i in range(len(parts))) + '];\n')
    for key in ['sha256', 'bytes', 'files']:
        manifest.pop(key, None)
    files = [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(directory.glob('*.bin'))]
    manifest.update(version=3, codec='zlib: f64 vertices and integer surface areas; hierarchical index',
        parts=len(parts), blocks=count, triangles=triangles, total_weight_units=total,
        root_index_bytes=len(root), bytes=sum(f['bytes'] for f in files),
        max_compressed_part_bytes=max(v[1] for v in parts), files=files,
        selection_weights='Nearest GRS80 surface integral on each unchanged triangle, 1e8 units/m2, minimum one unit',
        target_measure='GRS80 surface area on the unchanged piecewise-linear EPSG:4326 mask',
        uniformity='One u64 partitioned by surface mass; residual rank area quantile and digitally shifted bit reversal; inverse GRS80 surface CDF; binary64 coordinates; no supplemental entropy, PRNG or rejection')
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def upgrade(source, flat, directory):
    root = (source / 'index.bin').read_bytes()
    magic, parts, triangles, record, _, _ = struct.unpack_from('<8s4IQ', root)
    if magic != b'SRGKR002':
        raise ValueError('Expected v2 migration source')
    manifest = json.loads((source / 'manifest.json').read_text())
    blocks = int(manifest['blocks'])
    rows, running = [], 0
    geometry = hashlib.sha256()
    with flat.open('w+b') as output:
        output.seek(32 + blocks * 24)
        for part in range(parts):
            payload = (source / f'part-{part:03}.bin').read_bytes()
            _, count, _ = struct.unpack_from('<8sII', payload)
            for local in range(count):
                _, offset, length = struct.unpack_from('<QII', payload, 16 + local * 16)
                raw = zlib.decompress(payload[offset:offset + length])
                nv, nt = struct.unpack_from('<HH', raw)
                stop = 4 + nv * 16
                vertices = np.frombuffer(np.frombuffer(raw[4:stop], dtype='u1').reshape(8, nv * 2).T.copy().tobytes(), dtype='<f8').reshape(nv, 2)
                records = np.frombuffer(raw[stop:], dtype='u1').reshape(14, nt).T.copy()
                ids = np.ndarray((nt, 3), dtype='<u2', buffer=records, offset=8, strides=(14, 2))
                geometry.update(raw[:stop] + records[:, 8:].tobytes())
                weights = surface_weights(vertices[ids])
                records[:, :8] = weights.astype('<u8').view('u1').reshape(nt, 8)
                running += sum(map(int, weights))
                compressed = zlib.compress(raw[:stop] + records.T.tobytes(), 9)
                rows.append((running - 1, output.tell(), nt, len(compressed)))
                output.write(compressed)
            if part % 20 == 0:
                print(json.dumps({'parts': part, 'total': parts}), flush=True)
        output.seek(0)
        output.write(struct.pack('<8s4IQ', b'SRGKR003', blocks, triangles, record, 256, running))
        for row in rows:
            output.write(struct.pack('<QQII', *row))
    if geometry.hexdigest() != manifest['geometry_payload_sha256']:
        raise ValueError('Geometry payload changed')
    manifest.update(source_v2_index_sha256=hashlib.sha256(root).hexdigest(),
        geometry_payload_sha256=geometry.hexdigest(), geometry_bits_preserved=True)
    result = pack_flat(flat, manifest, directory)
    print(json.dumps({k: result[k] for k in ['version','blocks','triangles','parts','bytes','root_index_bytes','geometry_payload_sha256']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--flat', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    upgrade(args.source, args.flat, args.output)

