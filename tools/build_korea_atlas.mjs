import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {inflateSync} from 'node:zlib';

const ROOT_HASH = '8cb20a5c0da9c3d6225efb0323e24a4f992eb3029a772da3a71a55d3f503bf17';
const SOURCE_HASH = '5060f88241e2d80b87f052b4b83296ef59cd7f4ee066a9330bc2ea580bc55428';
const ATLAS_HASHES = new Map([
    [4, 'f3005c68a3ce34d2816acfd4e09e5f6c3fd29e925d2ba62f2d152327d5df4f63'],
    [8, 'a55dcf2e37a4500d6c30de3f5dad42131758415bcfafac79e88d76fb88b0e84d'],
]);

function requireValid(condition, message) {
    if (!condition) throw new Error(message);
}

function sha256(bytes) {
    return createHash('sha256').update(bytes).digest('hex');
}

function argumentsFrom(argv) {
    let boundary = null;
    let output = null;
    let subdivision = null;
    for (let index = 0; index < argv.length; index++) {
        const argument = argv[index];
        if (argument === '--output' && index + 1 < argv.length) {
            output = argv[++index];
        } else if (argument === '--subdivision' && index + 1 < argv.length) {
            subdivision = Number(argv[++index]);
        } else if (!argument.startsWith('-') && boundary === null) {
            boundary = argument;
        } else {
            throw new Error('Unexpected argument: ' + argument);
        }
    }
    requireValid(boundary && output && ATLAS_HASHES.has(subdivision),
        'Usage: node tools/build_korea_atlas.mjs <boundary-dir> --subdivision 4|8 --output <file>');
    return {boundary, output, subdivision};
}

function parseBoundary(directory) {
    const root = readFileSync(join(directory, 'index.bin'));
    requireValid(sha256(root) === ROOT_HASH && root.subarray(0, 8).toString() === 'SRGKR004',
        'Original boundary root identity mismatch');
    const get32 = (bytes, offset) => bytes.readUInt32LE(offset);
    const tileCount = get32(root, 12);
    const partCount = get32(root, 16);
    const pageCount = get32(root, 20);
    const columns = get32(root, 36);
    const rows = get32(root, 40);
    const scale = get32(root, 44);
    requireValid(get32(root, 8) === 128 && tileCount === columns * rows
        && partCount > 0 && partCount <= 4096 && pageCount > 0
        && root.length === 128 + tileCount * 12 + partCount * 48,
    'Original boundary root layout mismatch');
    requireValid(root.subarray(96, 128).toString('hex') === SOURCE_HASH,
        'Original boundary source identity mismatch');
    const west = root.readDoubleLE(48);
    const south = root.readDoubleLE(56);
    const tiles = [];
    let nextPage = 0;
    for (let tile = 0; tile < tileCount; tile++) {
        const offset = 128 + tile * 12;
        const state = get32(root, offset);
        const first = get32(root, offset + 4);
        const count = get32(root, offset + 8);
        requireValid(state <= 2 && first === nextPage && (count > 0) === (state === 2),
            'Original boundary tile index mismatch: ' + tile);
        tiles.push({state, first, count});
        nextPage += count;
    }
    requireValid(nextPage === pageCount, 'Original boundary tile page count mismatch');
    const table = [];
    const parts = [];
    nextPage = 0;
    for (let part = 0; part < partCount; part++) {
        const offset = 128 + tileCount * 12 + part * 48;
        const first = get32(root, offset);
        const count = get32(root, offset + 4);
        const size = get32(root, offset + 8);
        const digest = root.subarray(offset + 16, offset + 48).toString('hex');
        const fileName = 'part-' + String(part).padStart(3, '0') + '.bin';
        const bytes = readFileSync(join(directory, fileName));
        requireValid(first === nextPage && count > 0 && size <= 524288
            && bytes.length === size && sha256(bytes) === digest
            && bytes.subarray(0, 8).toString() === 'SRGKP004'
            && get32(bytes, 8) === first && get32(bytes, 12) === count
            && get32(bytes, 16) === size && get32(bytes, 20) === 0
            && bytes.subarray(24, 56).toString('hex') === SOURCE_HASH
            && bytes.readBigUInt64LE(56) === 0n,
        'Original boundary part invalid: ' + fileName);
        let nextOffset = 64 + count * 12;
        for (let page = 0; page < count; page++) {
            const location = 64 + page * 12;
            const start = get32(bytes, location);
            const length = get32(bytes, location + 4);
            const rawLength = get32(bytes, location + 8);
            requireValid(start === nextOffset && length > 0 && rawLength >= 32
                && rawLength <= 32768 && start + length <= bytes.length,
            'Original boundary page table invalid: ' + (first + page));
            table.push({part, start, length, rawLength});
            nextOffset += length;
        }
        requireValid(nextOffset === bytes.length, 'Original boundary part trailing data: ' + fileName);
        parts.push(bytes);
        nextPage += count;
    }
    requireValid(nextPage === pageCount, 'Original boundary part page count mismatch');
    return {tiles, parts, table, columns, rows, scale, west, south, tileCount};
}

function loadTilePages(boundary, tile) {
    const {first, count} = boundary.tiles[tile];
    const decoded = [];
    for (let number = first; number < first + count; number++) {
        const {part, start, length, rawLength} = boundary.table[number];
        const raw = inflateSync(boundary.parts[part].subarray(start, start + length), {
            maxOutputLength: rawLength + 1,
        });
        requireValid(raw.length === rawLength
            && raw.subarray(0, 8).toString() === 'SRGKB004'
            && raw.readUInt32LE(8) === tile && raw.readUInt32LE(12) === number,
        'Original boundary page invalid: ' + number);
        const kind = raw.readUInt32LE(16);
        const count = raw.readUInt32LE(20);
        if (kind === 0) {
            requireValid(count <= 4092 && raw.length === 32 + count * 8,
                'Original boundary parity page invalid: ' + number);
            const events = [];
            for (let i = 0; i < count; i++) events.push(raw.readDoubleLE(32 + i * 8));
            decoded.push({kind, base: raw.readUInt32LE(24), events});
        } else {
            requireValid(kind === 1 && count > 0 && count <= 1024
                && raw.length === 40 + (count + 1) * 16,
            'Original boundary ring page invalid: ' + number);
            const words = (count + 1) * 2;
            const coordinates = Buffer.allocUnsafe(words * 8);
            for (let word = 0; word < words; word++) {
                for (let lane = 0; lane < 8; lane++) {
                    coordinates[word * 8 + lane] = raw[40 + lane * words + word];
                }
            }
            const points = [];
            for (let i = 0; i <= count; i++) {
                points.push([coordinates.readDoubleLE(i * 16),
                    coordinates.readDoubleLE(i * 16 + 8)]);
            }
            decoded.push({kind, points});
        }
    }
    requireValid(decoded.length > 1 && decoded[0].kind === 0,
        'Original boundary mixed tile page ordering invalid: ' + tile);
    return decoded;
}

function segmentHitsRectangle(ax, ay, bx, by, west, south, east, north) {
    if (Math.min(ax, bx) > east || Math.max(ax, bx) < west
        || Math.min(ay, by) > north || Math.max(ay, by) < south) return false;
    const dx = bx - ax;
    const dy = by - ay;
    const crosses = [
        dx * (south - ay) - dy * (west - ax),
        dx * (south - ay) - dy * (east - ax),
        dx * (north - ay) - dy * (west - ax),
        dx * (north - ay) - dy * (east - ax),
    ];
    return !crosses.every(value => value > 0) && !crosses.every(value => value < 0);
}

function insideCell(longitude, latitude, parityPages, segments) {
    let parity = false;
    for (const page of parityPages) {
        let value = page.base;
        for (const event of page.events) {
            if (event > latitude) break;
            value ^= 1;
        }
        parity = parity !== (value !== 0);
    }
    for (const [ax, ay, bx, by] of segments) {
        if ((ay > latitude) !== (by > latitude)) {
            const determinant = (bx - ax) * (latitude - ay)
                - (by - ay) * (longitude - ax);
            if (by > ay ? determinant > 0 : determinant < 0) parity = !parity;
        }
    }
    return parity;
}

function buildAtlas(boundary, subdivision) {
    const mixedTiles = [];
    for (let tile = 0; tile < boundary.tileCount; tile++) {
        if (boundary.tiles[tile].state === 2) mixedTiles.push(tile);
    }
    const area = subdivision * subdivision;
    const states = Buffer.alloc(mixedTiles.length * area / 4);
    const counts = [0, 0, 0];
    const step = 1 / (boundary.scale * subdivision);
    for (let ordinal = 0; ordinal < mixedTiles.length; ordinal++) {
        const tile = mixedTiles[ordinal];
        const column = tile % boundary.columns;
        const row = Math.floor(tile / boundary.columns);
        const west = boundary.west + column / boundary.scale;
        const south = boundary.south + row / boundary.scale;
        const pages = loadTilePages(boundary, tile);
        const parityPages = pages.filter(page => page.kind === 0);
        const segments = [];
        for (const page of pages) {
            if (page.kind !== 1) continue;
            for (let edge = 1; edge < page.points.length; edge++) {
                segments.push([...page.points[edge - 1], ...page.points[edge]]);
            }
        }
        for (let y = 0; y < subdivision; y++) {
            for (let x = 0; x < subdivision; x++) {
                const x0 = west + x * step;
                const y0 = south + y * step;
                const x1 = x0 + step;
                const y1 = y0 + step;
                const hasBoundary = segments.some(([ax, ay, bx, by]) =>
                    segmentHitsRectangle(ax, ay, bx, by, x0, y0, x1, y1));
                const state = hasBoundary ? 2
                    : Number(insideCell((x0 + x1) / 2, (y0 + y1) / 2,
                        parityPages, segments));
                counts[state]++;
                const index = ordinal * area + y * subdivision + x;
                states[index >>> 2] |= state << ((index & 3) * 2);
            }
        }
    }
    const header = Buffer.alloc(96);
    header.write('SRGKA001');
    header.writeUInt32LE(96, 8);
    header.writeUInt32LE(subdivision, 12);
    header.writeUInt32LE(mixedTiles.length, 16);
    header.writeUInt32LE(boundary.tileCount, 20);
    Buffer.from(ROOT_HASH, 'hex').copy(header, 24);
    Buffer.from(SOURCE_HASH, 'hex').copy(header, 56);
    header.writeUInt32LE(states.length, 88);
    return {blob: Buffer.concat([header, states]), counts};
}

function main() {
    const {boundary, output, subdivision} = argumentsFrom(process.argv.slice(2));
    const {blob, counts} = buildAtlas(parseBoundary(boundary), subdivision);
    const digest = sha256(blob);
    requireValid(digest === ATLAS_HASHES.get(subdivision),
        'Generated atlas does not match independently certified geometry');
    writeFileSync(output, blob);
    process.stdout.write(JSON.stringify({file: output, subdivision, bytes: blob.length,
        sha256: digest, states_outside_inside_mixed: counts,
        root_sha256: ROOT_HASH, source_sha256: SOURCE_HASH}) + '\n');
}

main();
