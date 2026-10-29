import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, box


def require(condition, message):
    if not condition:
        raise ValueError(message)


class PackedBoundary:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.root = (self.directory / 'index.bin').read_bytes()
        require(128 <= len(self.root) <= 8388608, 'Invalid root size')
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        require(hashlib.sha256(self.root).hexdigest() == manifest['root_index_sha256'],
                'Root SHA-256 mismatch')
        fields = struct.unpack_from('<8s10I6d32s', self.root)
        (magic, header, tile_count, part_count, page_count, self.polygon_count,
         self.ring_count, self.vertices, self.columns, self.rows, self.scale,
         self.grid_west, self.grid_south, *tail) = fields
        self.bounds = tuple(tail[:4])
        self.digest = tail[4]
        require(magic == b'SRGKR004' and header == 128, 'Invalid root version')
        require(0 < tile_count == self.columns * self.rows <= 262144, 'Invalid tile count')
        require(0 < part_count <= 4096 and 0 < page_count <= 1000000, 'Invalid part/page count')
        require(self.scale in (1, 2, 4, 8, 16, 32, 64, 128, 256), 'Invalid tile scale')
        require(len(self.root) == 128 + tile_count * 12 + part_count * 48, 'Invalid root length')
        west, south, east, north = self.bounds
        require(124 <= west < east <= 132 and 32 <= south < north <= 40, 'Invalid root bounds')
        require(self.grid_west == math.floor(west * self.scale) / self.scale
                and self.grid_south == math.floor(south * self.scale) / self.scale,
                'Invalid grid origin')
        require(self.columns == math.ceil(east * self.scale) - math.floor(west * self.scale)
                and self.rows == math.ceil(north * self.scale) - math.floor(south * self.scale),
                'Invalid grid dimensions')
        self.tiles = [struct.unpack_from('<3I', self.root, 128 + tile * 12)
                      for tile in range(tile_count)]
        first = 0
        for state, start, count in self.tiles:
            require(state <= 2 and start == first and (count > 0) == (state == 2),
                    'Invalid tile page range')
            first += count
        require(first == page_count, 'Incomplete tile page index')
        self.parts = []
        self.page_entries = []
        first = 0
        for number in range(part_count):
            entry = 128 + tile_count * 12 + number * 48
            begin, count, size, reserved, digest = struct.unpack_from('<4I32s', self.root, entry)
            require(begin == first and count > 0 and reserved == 0, 'Invalid part range')
            blob = (self.directory / f'part-{number:03}.bin').read_bytes()
            require(64 < len(blob) == size <= 524288, 'Invalid part size')
            require(hashlib.sha256(blob).digest() == digest, 'Part SHA-256 mismatch')
            magic, start, length, declared, reserved, source, padding = struct.unpack_from('<8s4I32sQ', blob)
            require((magic, start, length, declared, reserved, source, padding)
                    == (b'SRGKP004', begin, count, size, 0, self.digest, 0), 'Invalid part header')
            offset = 64 + count * 12
            for local in range(count):
                position, length, raw_size = struct.unpack_from('<3I', blob, 64 + local * 12)
                require(position == offset and length > 0 and 32 <= raw_size <= 32768,
                        'Invalid page table')
                offset += length
                require(offset <= len(blob), 'Page exceeds part size')
                self.page_entries.append((number, position, length, raw_size))
            require(offset == len(blob), 'Trailing bytes in part')
            self.parts.append(blob)
            first += count
        require(first == page_count, 'Incomplete part page index')
        self.decoded_tiles = {}

    def page(self, number, tile):
        part, offset, size, raw_size = self.page_entries[number]
        inflater = zlib.decompressobj()
        raw = inflater.decompress(self.parts[part][offset:offset + size], raw_size + 1)
        require(len(raw) == raw_size and inflater.eof and not inflater.unused_data
                and not inflater.unconsumed_tail, 'Invalid compressed page')
        magic, owner, page, kind, count, field_a, field_b = struct.unpack_from('<8s6I', raw)
        require((magic, owner, page) == (b'SRGKB004', tile, number), 'Invalid page identity')
        if kind == 0:
            require(count <= 4092 and len(raw) == 32 + count * 8
                    and field_a <= 1 and field_b == 0, 'Invalid parity page')
            events = np.frombuffer(raw, '<f8', count, 32)
            row = tile // self.columns
            south = self.grid_south + row / self.scale
            north = south + 1 / self.scale
            require(np.all(np.isfinite(events)) and np.all(events > south)
                    and np.all(events <= north) and np.all(np.diff(events) > 0),
                    'Invalid parity events')
            return kind, field_a, events
        require(kind == 1 and 0 < count <= 1024 and field_a < self.ring_count
                and field_b < self.polygon_count and len(raw) == 40 + (count + 1) * 16,
                'Invalid chain page')
        first, hole = struct.unpack_from('<2I', raw, 32)
        require(hole <= 1 and first + count < self.vertices, 'Invalid chain metadata')
        shuffled = np.frombuffer(raw, 'u1', offset=40).reshape(8, (count + 1) * 2)
        coordinates = np.frombuffer(shuffled.T.copy().tobytes(), '<f8').reshape(count + 1, 2)
        west, south, east, north = self.bounds
        require(np.all(np.isfinite(coordinates))
                and np.all((coordinates[:, 0] >= west) & (coordinates[:, 0] <= east))
                and np.all((coordinates[:, 1] >= south) & (coordinates[:, 1] <= north)),
                'Invalid chain coordinate')
        return kind, field_a, field_b, first, hole, coordinates

    def tile_pages(self, tile):
        if tile not in self.decoded_tiles:
            state, first, count = self.tiles[tile]
            require(state == 2, 'Non-mixed tile has no pages')
            pages = [self.page(number, tile) for number in range(first, first + count)]
            require(pages[0][0] == 0, 'Missing initial parity page')
            geometry = False
            previous = -math.inf
            for offset, page in enumerate(pages):
                if page[0] == 0:
                    require(not geometry and (offset == 0 or page[1] == 0), 'Parity page ordering error')
                    events = page[2]
                    if len(events):
                        require(events[0] > previous, 'Parity events not globally sorted')
                        previous = events[-1]
                else:
                    geometry = True
            require(geometry, 'Mixed tile has no chain pages')
            self.decoded_tiles[tile] = pages
        return self.decoded_tiles[tile]

    def covers(self, longitude, latitude):
        west, south, east, north = self.bounds
        if not (west <= longitude <= east and south <= latitude <= north):
            return False
        column = min(self.columns - 1, math.floor((longitude - self.grid_west) * self.scale))
        row = min(self.rows - 1, math.floor((latitude - self.grid_south) * self.scale))
        tile = row * self.columns + column
        state = self.tiles[tile][0]
        if state != 2:
            return state == 1
        parity = False
        boundary = False
        for page in self.tile_pages(tile):
            if page[0] == 0:
                parity ^= bool(page[1] ^ (int(np.searchsorted(page[2], latitude, side='right')) % 2))
                continue
            coordinates = page[-1]
            for start, end in zip(coordinates[:-1], coordinates[1:], strict=True):
                if not min(start[1], end[1]) <= latitude <= max(start[1], end[1]):
                    continue
                determinant = orientation(start, end, longitude, latitude)
                if determinant == 0 and min(start[0], end[0]) <= longitude <= max(start[0], end[0]):
                    boundary = True
                if (start[1] > latitude) != (end[1] > latitude):
                    parity ^= determinant > 0 if end[1] > start[1] else determinant < 0
        return boundary or parity


def orientation(start, end, longitude, latitude):
    horizontal = float(end[0] - start[0])
    vertical = float(end[1] - start[1])
    left = horizontal * (latitude - float(start[1]))
    right = vertical * (longitude - float(start[0]))
    result = left - right
    if abs(result) > (abs(left) + abs(right)) * 1e-14:
        return result
    return ((Fraction(float(end[0])) - Fraction(float(start[0])))
            * (Fraction(latitude) - Fraction(float(start[1])))
            - (Fraction(float(end[1])) - Fraction(float(start[1])))
            * (Fraction(longitude) - Fraction(float(start[0]))))


def validate_reconstruction(directory, source):
    boundary = PackedBoundary(directory)
    fragments = [[] for _ in range(boundary.ring_count)]
    owners = {}
    for tile, (state, _, _) in enumerate(boundary.tiles):
        if state != 2:
            row, column = divmod(tile, boundary.columns)
            west = boundary.grid_west + column / boundary.scale
            south = boundary.grid_south + row / boundary.scale
            cell = box(west, south, west + 1 / boundary.scale, south + 1 / boundary.scale)
            require(source.covers(cell) if state == 1 else source.disjoint(cell),
                    'Unsafe tile classification')
            continue
        for page in boundary.tile_pages(tile):
            if page[0] == 1:
                _, ring, polygon, first, hole, vertices = page
                require(ring not in owners or owners[ring] == (polygon, hole),
                        'Conflicting ring ownership')
                owners[ring] = (polygon, hole)
                fragments[ring].append((first, vertices))
    grouped = [[] for _ in range(boundary.polygon_count)]
    total_vertices = 0
    for ring, pieces in enumerate(fragments):
        require(bool(pieces), 'Missing source ring')
        count = max(first + len(vertices) for first, vertices in pieces)
        reconstructed = np.empty((count, 2), dtype='<f8')
        covered = np.zeros(count, dtype=bool)
        for first, vertices in pieces:
            end = first + len(vertices)
            existing = covered[first:end]
            require(np.array_equal(reconstructed[first:end][existing].view('u8'),
                                   vertices[existing].view('u8')), 'Conflicting source vertex copies')
            reconstructed[first:end] = vertices
            covered[first:end] = True
        require(np.all(covered) and np.array_equal(reconstructed[0], reconstructed[-1]),
                'Incomplete reconstructed source ring')
        polygon, hole = owners[ring]
        grouped[polygon].append((hole, reconstructed))
        total_vertices += count
    polygons = []
    for rings in grouped:
        exterior = [vertices for hole, vertices in rings if not hole]
        holes = [vertices for hole, vertices in rings if hole]
        require(len(exterior) == 1, 'Missing or duplicate exterior ring')
        polygons.append(Polygon(exterior[0], holes))
    reconstructed = MultiPolygon(polygons)
    source_multi = MultiPolygon([source]) if source.geom_type == 'Polygon' else source
    source_coordinates = shapely.get_coordinates(source_multi)
    reconstructed_coordinates = shapely.get_coordinates(reconstructed)
    preserved = np.array_equal(source_coordinates.view('u8'), reconstructed_coordinates.view('u8'))
    require(preserved, 'Original coordinate bits changed')
    require(total_vertices == boundary.vertices, 'Source vertex count mismatch')
    print(json.dumps({'verification': 'global_validity', 'vertices': total_vertices}), flush=True)
    source_valid = bool(source_multi.is_valid)
    reconstructed_valid = bool(reconstructed.is_valid)
    require(source_valid and reconstructed_valid, 'Source or reconstructed mask is invalid')
    exact = bool(shapely.equals_exact(source_multi, reconstructed, 0))
    require(exact, 'Reconstructed mask coordinates differ from source')
    checked = 0
    for index, (original, restored) in enumerate(zip(source_multi.geoms, reconstructed.geoms, strict=True)):
        require(original.equals(restored), f'Component {index} is not topologically equal')
        require(original.symmetric_difference(restored).is_empty,
                f'Component {index} has a nonempty symmetric difference')
        checked += 1
        if checked % 1000 == 0:
            print(json.dumps({'verification': 'component_equality', 'checked': checked}), flush=True)
    return {
        'source_valid': source_valid,
        'reconstructed_valid': reconstructed_valid,
        'equals_exact_zero_tolerance': exact,
        'equals': True,
        'symmetric_difference_empty': True,
        'symmetric_difference_area_deg2': 0.0,
        'topological_comparison_scope': 'Every matched polygon component; exact component equality implies union equality',
        'geos_component_pairs_checked': checked,
        'polygon_components': len(polygons),
        'rings': boundary.ring_count,
        'vertices': total_vertices,
        'all_source_vertices_bit_preserved': preserved,
        'all_tile_classifications_verified': True,
        'all_part_sha256_and_zlib_checksums_verified': True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    from build_korea_boundary import load_source
    result = validate_reconstruction(args.directory, load_source(args.source))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
