import hashlib
import os
from pathlib import Path
import struct
import zlib


MAX_PART_BYTES = 524288
MAX_PAGE_BYTES = 32768
MAX_CHAIN_EDGES = 1024
MAX_PARITY_EVENTS = (MAX_PAGE_BYTES - 32) // 8
ROOT_HEADER_BYTES = 128
PART_HEADER_BYTES = 64


class BoundaryPacker:
    def __init__(self, directory, source_digest):
        self.directory = Path(directory)
        self.source_digest = bytes.fromhex(source_digest)
        if len(self.source_digest) != 32:
            raise ValueError('Invalid source digest')
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise ValueError('Boundary output directory must be empty')
        self.parts = []
        self.pages = []
        self.page_count = 0
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.max_page_bytes = 0
        self.part_bytes = PART_HEADER_BYTES

    def append(self, tile, kind, count, field_a, field_b, payload):
        raw = struct.pack(
            '<8s6I', b'SRGKB004', tile, self.page_count,
            kind, count, field_a, field_b,
        ) + payload
        if len(raw) > MAX_PAGE_BYTES:
            raise ValueError('Boundary page exceeds raw byte limit')
        compressed = zlib.compress(raw, 9)
        if self.part_bytes + 12 + len(compressed) > MAX_PART_BYTES:
            self.flush()
        self.pages.append((compressed, len(raw)))
        self.part_bytes += 12 + len(compressed)
        self.page_count += 1
        self.raw_bytes += len(raw)
        self.compressed_bytes += len(compressed)
        self.max_page_bytes = max(self.max_page_bytes, len(raw))
        if self.page_count > 1000000:
            raise ValueError('Boundary page count limit exceeded')

    def flush(self):
        if not self.pages:
            return
        first = self.page_count - len(self.pages)
        offset = PART_HEADER_BYTES + len(self.pages) * 12
        table = []
        payloads = []
        for compressed, raw_bytes in self.pages:
            table.append(struct.pack('<3I', offset, len(compressed), raw_bytes))
            payloads.append(compressed)
            offset += len(compressed)
        header = struct.pack(
            '<8s4I32sQ', b'SRGKP004', first, len(self.pages),
            offset, 0, self.source_digest, 0,
        )
        blob = header + b''.join(table + payloads)
        if len(blob) != self.part_bytes or len(blob) > MAX_PART_BYTES:
            raise ValueError('Invalid boundary part size')
        path = self.directory / f'part-{len(self.parts):03}.bin'
        path.write_bytes(blob)
        self.parts.append((first, len(self.pages), len(blob), hashlib.sha256(blob).digest()))
        if len(self.parts) > 4096:
            raise ValueError('Boundary part count limit exceeded')
        self.pages = []
        self.part_bytes = PART_HEADER_BYTES

    def finish(self, tiles, columns, rows, scale, grid_west, grid_south,
               bounds, polygon_count, ring_count, source_vertices):
        self.flush()
        if not self.parts or len(tiles) != columns * rows or len(tiles) > 262144:
            raise ValueError('Invalid boundary index dimensions')
        header = struct.pack(
            '<8s10I6d32s', b'SRGKR004', ROOT_HEADER_BYTES,
            len(tiles), len(self.parts), self.page_count,
            polygon_count, ring_count, source_vertices,
            columns, rows, scale, grid_west, grid_south,
            *bounds, self.source_digest,
        )
        records = [struct.pack('<3I', *tile) for tile in tiles]
        parts = [struct.pack('<4I32s', first, count, size, 0, digest)
                 for first, count, size, digest in self.parts]
        root = header + b''.join(records + parts)
        if len(header) != ROOT_HEADER_BYTES or len(root) > 8 * 1024 * 1024:
            raise ValueError('Invalid root index size')
        (self.directory / 'index.bin').write_bytes(root)
        return root


def write_rust_module(directory, module, root):
    module = Path(module)
    module.parent.mkdir(parents=True, exist_ok=True)
    prefix = Path(os.path.relpath(directory, module.parent)).as_posix()
    parts = sorted(Path(directory).glob('part-*.bin'))
    digest = ', '.join(str(value) for value in hashlib.sha256(root).digest())
    text = f'pub(super) const INDEX_SHA256: [u8; 32] = [{digest}];\n'
    text += f'pub(super) static INDEX: &[u8] = include_bytes!("{prefix}/index.bin");\n'
    text += f'pub(super) static PARTS: [&[u8]; {len(parts)}] = [\n'
    text += ''.join(f'    include_bytes!("{prefix}/{path.name}"),\n' for path in parts)
    module.write_text(text + '];\n')
