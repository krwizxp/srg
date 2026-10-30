const INFLATE_CODE_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
const INFLATE_DISTANCE_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const INFLATE_DISTANCE_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const INFLATE_LENGTH_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const INFLATE_LENGTH_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];

function inflateRequire(condition, message) {
    if (!condition) throw new Error(message);
}

class InflateBits {
    constructor(bytes) {
        this.bytes = bytes;
        this.position = 0;
        this.length = bytes.length * 8;
    }

    peek(count) {
        inflateRequire(this.position + count <= this.length, "경계 압축 데이터가 잘렸습니다.");
        const index = Math.floor(this.position / 8);
        return ((this.bytes[index] | this.bytes[index + 1] << 8 | this.bytes[index + 2] << 16)
            >>> (this.position % 8)) & ((1 << count) - 1);
    }

    read(count) {
        const value = this.peek(count);
        this.position += count;
        return value;
    }
}

function inflateHuffman(lengths, complete = false) {
    inflateRequire(lengths.length <= 288, "경계 압축 코드 개수 오류");
    const counts = new Uint16Array(16);
    const symbols = new Uint16Array(288);
    const offsets = new Uint16Array(16);
    const quick = new Uint16Array(512);
    for (const length of lengths) {
        inflateRequire(length <= 15, "경계 압축 코드 길이 오류");
        counts[length]++;
    }
    let remaining = 1;
    for (let length = 1; length < 16; length++) {
        remaining = remaining * 2 - counts[length];
        inflateRequire(remaining >= 0, "경계 압축 코드가 중복됩니다.");
        if (length > 1) offsets[length] = offsets[length - 1] + counts[length - 1];
    }
    inflateRequire(remaining === 0 || (!complete && counts[0] + counts[1] === lengths.length), "경계 압축 코드가 불완전합니다.");
    for (let symbol = 0; symbol < lengths.length; symbol++) {
        const length = lengths[symbol];
        if (length !== 0) symbols[offsets[length]++] = symbol;
    }
    let first = 0;
    let index = 0;
    for (let length = 1; length < 16; length++) {
        for (let offset = 0; offset < counts[length]; offset++) {
            if (length <= 9) {
                let code = first + offset;
                let reversed = 0;
                for (let bit = 0; bit < length; bit++) {
                    reversed = (reversed << 1) | (code & 1);
                    code >>>= 1;
                }
                const packed = (length << 9) | symbols[index + offset];
                for (let slot = reversed; slot < 512; slot += 1 << length) quick[slot] = packed;
            }
        }
        index += counts[length];
        first = (first + counts[length]) * 2;
    }
    return {counts, symbols, quick};
}

function inflateSymbol(tree, bits) {
    if (bits.position + 9 <= bits.length) {
        const index = bits.position >>> 3;
        const packed = tree.quick[((bits.bytes[index] | bits.bytes[index + 1] << 8)
            >>> (bits.position & 7)) & 511];
        if (packed !== 0) {
            bits.position += packed >>> 9;
            return packed & 511;
        }
    }
    let code = 0;
    let first = 0;
    let index = 0;
    for (let length = 1; length < 16; length++) {
        code |= bits.read(1);
        const count = tree.counts[length];
        if (code < first + count) return tree.symbols[index + code - first];
        index += count;
        first = (first + count) * 2;
        code <<= 1;
    }
    throw new Error("경계 압축 허프먼 코드 오류");
}

function inflateDynamic(bits) {
    const literalCount = bits.read(5) + 257;
    const distanceCount = bits.read(5) + 1;
    const codeCount = bits.read(4) + 4;
    inflateRequire(literalCount <= 286, "경계 압축 코드 개수 오류");
    const codeLengths = new Uint8Array(19);
    for (let index = 0; index < codeCount; index++) codeLengths[INFLATE_CODE_ORDER[index]] = bits.read(3);
    const codes = inflateHuffman(codeLengths, true);
    const total = literalCount + distanceCount;
    const lengths = new Uint8Array(total);
    let filled = 0;
    while (filled < total) {
        const symbol = inflateSymbol(codes, bits);
        let value;
        let count;
        if (symbol <= 15) {
            value = symbol;
            count = 1;
        } else if (symbol === 16) {
            inflateRequire(filled > 0, "경계 압축 코드 반복 오류");
            value = lengths[filled - 1];
            count = bits.read(2) + 3;
        } else if (symbol === 17 || symbol === 18) {
            value = 0;
            count = symbol === 17 ? bits.read(3) + 3 : bits.read(7) + 11;
        } else {
            throw new Error("경계 압축 코드 길이 오류");
        }
        const next = filled + count;
        inflateRequire(next <= total, "경계 압축 코드 반복 한도 오류");
        lengths.fill(value, filled, next);
        filled = next;
    }
    inflateRequire(lengths[256] !== 0, "경계 압축 종료 코드 오류");
    return [inflateHuffman(lengths.subarray(0, literalCount)), inflateHuffman(lengths.subarray(literalCount))];
}

let inflateFixedTrees;

function inflate(input, maximum = 32768) {
    inflateRequire(input instanceof Uint8Array && input.length >= 6, "경계 압축 길이 오류");
    inflateRequire(Number.isSafeInteger(maximum) && maximum >= 0 && maximum <= 32768, "경계 압축 출력 한도 오류");
    inflateRequire((input[0] & 15) === 8 && (input[0] >>> 4) <= 7 && (input[1] & 32) === 0
        && ((input[0] << 8) | input[1]) % 31 === 0, "경계 압축 형식 오류");
    const tail = input.length - 4;
    const bits = new InflateBits(input.subarray(2, tail));
    const output = new Uint8Array(maximum);
    let written = 0;
    for (;;) {
        const finalBlock = bits.read(1) !== 0;
        const kind = bits.read(2);
        if (kind === 0) {
            bits.position = Math.ceil(bits.position / 8) * 8;
            const length = bits.read(16);
            inflateRequire((length ^ bits.read(16)) === 65535, "경계 압축 저장 블록 오류");
            inflateRequire(written + length <= maximum, "경계 압축 출력 한도 오류");
            for (let index = 0; index < length; index++) output[written++] = bits.read(8);
        } else if (kind === 1 || kind === 2) {
            if (kind === 1 && !inflateFixedTrees) {
                const lengths = new Uint8Array(288).fill(8);
                lengths.fill(9, 144, 256);
                lengths.fill(7, 256, 280);
                inflateFixedTrees = [inflateHuffman(lengths), inflateHuffman(new Uint8Array(32).fill(5))];
            }
            const [literal, distance] = kind === 1 ? inflateFixedTrees : inflateDynamic(bits);
            for (;;) {
                const symbol = inflateSymbol(literal, bits);
                if (symbol === 256) break;
                if (symbol < 256) {
                    inflateRequire(written < maximum, "경계 압축 출력 한도 오류");
                    output[written++] = symbol;
                    continue;
                }
                const index = symbol - 257;
                inflateRequire(index < INFLATE_LENGTH_BASE.length, "경계 압축 길이 코드 오류");
                const length = INFLATE_LENGTH_BASE[index] + bits.read(INFLATE_LENGTH_EXTRA[index]);
                const distanceIndex = inflateSymbol(distance, bits);
                inflateRequire(distanceIndex < INFLATE_DISTANCE_BASE.length, "경계 압축 거리 코드 오류");
                const backward = INFLATE_DISTANCE_BASE[distanceIndex] + bits.read(INFLATE_DISTANCE_EXTRA[distanceIndex]);
                inflateRequire(backward <= written, "경계 압축 참조 위치 오류");
                inflateRequire(written + length <= maximum, "경계 압축 출력 한도 오류");
                for (let index = 0; index < length; index++) {
                    output[written] = output[written - backward];
                    written++;
                }
            }
        } else {
            throw new Error("경계 압축 블록 형식 오류");
        }
        if (finalBlock) break;
    }
    inflateRequire(Math.ceil(bits.position / 8) === bits.bytes.length, "경계 압축 잔여 데이터 오류");
    let first = 1;
    let second = 0;
    for (let begin = 0; begin < written; begin += 1024) {
        const end = Math.min(begin + 1024, written);
        for (let index = begin; index < end; index++) {
            first += output[index];
            second += first;
        }
        first %= 65521;
        second %= 65521;
    }
    const checksum = ((input[tail] << 24) | (input[tail + 1] << 16) | (input[tail + 2] << 8) | input[tail + 3]) >>> 0;
    inflateRequire((((second << 16) | first) >>> 0) === checksum, "경계 압축 체크섬 오류");
    return output.subarray(0, written);
}

const DEGREES = 0.017453292519943295;
const ECCENTRICITY_SQUARED = 0.006694380022900787;
function productError(a, b, product) {
    const splitA = 134217729 * a;
    const ah = splitA - (splitA - a);
    const al = a - ah;
    const splitB = 134217729 * b;
    const bh = splitB - (splitB - b);
    const bl = b - bh;
    return ((ah * bh - product) + ah * bl + al * bh) + al * bl;
}
function orientation(start, end, point) {
    const a = end[0] - start[0];
    const b = end[1] - start[1];
    const c = point[0] - start[0];
    const d = point[1] - start[1];
    const left = a * d;
    const right = b * c;
    const approximate = left - right;
    if (Math.abs(approximate) > 3.3306690738754716e-16 * (Math.abs(left) + Math.abs(right))) {
        return approximate;
    }
    const expansion = [0, 0, 0, 0];
    let length = 0;
    for (let value of [productError(a, d, left), left, -productError(b, c, right), -right]) {
        let count = 0;
        const previous = expansion.slice();
        for (let index = 0; index < length; index++) {
            const entry = previous[index];
            const total = value + entry;
            const virtualValue = total - value;
            const error = (value - (total - virtualValue)) + (entry - virtualValue);
            if (error !== 0) {
                expansion[count] = error;
                count++;
            }
            value = total;
        }
        if (value !== 0 || count === 0) {
            expansion[count] = value;
            count++;
        }
        length = count;
    }
    let result = 0;
    for (let index = 0; index < length; index++) {
        result += expansion[index];
    }
    return result;
}
function cumulativeArea(latitude) {
    const delta = (latitude - 36) * DEGREES;
    const square = delta * delta;
    const sineDelta = delta * (1 + square * (-0.16666666666666666
        + square * (0.008333333333333333
            + square * (-0.0001984126984126984
                + square * (2.7557319223985893e-6
                    + square * -2.505210838544172e-8)))));
    const cosineDelta = 1 + square * (-0.5
        + square * (0.041666666666666664
            + square * (-0.001388888888888889
                + square * (2.48015873015873e-5
                    + square * (-2.755731922398589e-7
                        + square * 2.08767569878681e-9)))));
    const sine = 0.5877852522924731 * cosineDelta + 0.8090169943749475 * sineDelta;
    const squareEccentricSine = ECCENTRICITY_SQUARED * sine * sine;
    let polynomial = 9 / 17;
    for (const coefficient of [8 / 15, 7 / 13, 6 / 11, 5 / 9, 4 / 7, 3 / 5, 2 / 3, 1]) {
        polynomial = coefficient + squareEccentricSine * polynomial;
    }
    return sine * polynomial;
}
function candidate(value, bounds) {
    if (typeof value !== "bigint" || value < 0n || value > 0xffffffffffffffffn) {
        throw new Error("대한민국 등면적 좌표 입력 오류");
    }
    const [west, south, east, north] = bounds;
    if (!(bounds.length === 4 && bounds.every(Number.isFinite)
        && west >= -180 && east <= 180 && west < east
        && south >= 32 && north <= 40 && south < north)) {
        throw new Error("대한민국 등면적 좌표 범위 오류");
    }
    const vertical = (Number(value >> 32n) + 0.5) / 4294967296;
    const horizontal = (Number(value & 0xffffffffn) + 0.5) / 4294967296;
    const southArea = cumulativeArea(south);
    const target = southArea + vertical * (cumulativeArea(north) - southArea);
    let lower = south;
    let upper = north;
    for (let iteration = 0; iteration < 52; iteration++) {
        const middle = lower + (upper - lower) * 0.5;
        if (cumulativeArea(middle) < target) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    const latitude = lower + (upper - lower) * 0.5;
    const longitude = west + horizontal * (east - west);
    if (!(latitude > south && latitude < north && longitude > west && longitude < east)) {
        throw new Error("대한민국 등면적 좌표 역산 오류");
    }
    return [latitude, longitude];
}
const MAX_U64 = (1n << 64n) - 1n;
const INDEX_SHA256 = "8cb20a5c0da9c3d6225efb0323e24a4f992eb3029a772da3a71a55d3f503bf17";
const TILE_SHA256 = "492fdbeee7e7a64816d7ea90c57cb5773685fbf96e0b95911a8b7cdcfbf2ae8b";
const TILE_COUNT = 79113;
const SOURCE_SHA256 = "5060f88241e2d80b87f052b4b83296ef59cd7f4ee066a9330bc2ea580bc55428";
const MAX_INDEX_BYTES = 8388608;
const MAX_PART_BYTES = 524288;
const MAX_PAGE_BYTES = 32768;
const SUPPLEMENT_LIMIT = 1024;
let boundaryPromise;
let preparation;
let requestedPreparation;
const parts = {entries: new Map(), bytes: 0, pending: 0, entryLimit: 8,
    byteLimit: 4194304, pendingLimit: 8, reservation: MAX_PART_BYTES};
const pages = {entries: new Map(), bytes: 0, pending: 0, entryLimit: 4096,
    byteLimit: 4194304, pendingLimit: 256, reservation: MAX_PAGE_BYTES};
function requireBoundary(condition, message) {
    if (!condition) throw new Error(message);
}
function hexBytes(bytes) {
    return Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
}
function magic(bytes, expected) {
    return bytes.length >= 8 && new TextDecoder().decode(bytes.subarray(0, 8)) === expected;
}
async function boundedBytes(response, maximum) {
    requireBoundary(response.body !== null, "대한민국 경계 응답 본문 오류");
    const reader = response.body.getReader();
    const chunks = [];
    let size = 0;
    try {
        for (;;) {
            const {value, done} = await reader.read();
            if (done) break;
            size += value.length;
            requireBoundary(size <= maximum, "대한민국 경계 데이터 크기 한도를 초과했습니다.");
            chunks.push(value);
        }
    } catch (error) {
        await reader.cancel().catch(() => {});
        throw error;
    } finally {
        reader.releaseLock();
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.length;
    }
    return bytes;
}
async function download(name, maximum, digest) {
    requireBoundary(typeof globalThis.crypto?.subtle?.digest === "function",
        "대한민국 경계 무결성 검증을 지원하는 HTTPS 환경이 필요합니다.");
    requireBoundary(/^[0-9a-f]{64}$/.test(digest), "대한민국 경계 서명 정보 오류");
    const url = new URL("./data/korea-boundary-v4/" + name, import.meta.url);
    url.searchParams.set("sha256", digest);
    const response = await fetch(url, {signal: AbortSignal.timeout(30000)});
    requireBoundary(response.ok, "대한민국 경계 데이터 불러오기 실패: HTTP " + response.status);
    const bytes = await boundedBytes(response, maximum);
    const actual = hexBytes(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)));
    requireBoundary(actual === digest, "대한민국 경계 SHA-256 불일치");
    return bytes;
}
async function cached(cache, key, load, byteLength) {
    for (;;) {
        const present = cache.entries.get(key);
        if (present) {
            cache.entries.delete(key);
            cache.entries.set(key, present);
            return present.promise;
        }
        if (cache.entries.size < cache.entryLimit && cache.pending < cache.pendingLimit
            && cache.bytes + cache.reservation <= cache.byteLimit) break;
        const [oldestKey, oldest] = cache.entries.entries().next().value;
        if (oldest.settled) {
            cache.entries.delete(oldestKey);
            cache.bytes -= oldest.bytes;
        } else {
            await oldest.promise.catch(() => {});
        }
    }
    const entry = {settled: false, promise: null, bytes: cache.reservation};
    cache.bytes += entry.bytes;
    cache.pending++;
    entry.promise = Promise.resolve().then(load).then(value => {
        cache.pending--;
        const size = byteLength(value);
        cache.bytes += size - entry.bytes;
        entry.bytes = size;
        entry.settled = true;
        return value;
    }, error => {
        cache.pending--;
        cache.bytes -= entry.bytes;
        if (cache.entries.get(key) === entry) cache.entries.delete(key);
        throw error;
    });
    cache.entries.set(key, entry);
    return entry.promise;
}
async function loadBoundary() {
    if (!boundaryPromise) boundaryPromise = (async () => {
        const [bytes, tileBytes] = await Promise.all([
            download("index.bin", MAX_INDEX_BYTES, INDEX_SHA256),
            download("tile-select-v5.bin", 32768, TILE_SHA256),
        ]);
        requireBoundary(tileBytes.length === 32768,
            "대한민국 좌표 타일 인덱스 길이 오류");
        const tileIndex = new Uint32Array(TILE_COUNT);
        let selected = 0;
        for (let tile = 0; tile < 262144; tile++) {
            if (tileBytes[tile >>> 3] & (1 << (tile & 7))) {
                requireBoundary(selected < TILE_COUNT, "대한민국 좌표 타일 개수 오류");
                tileIndex[selected++] = tile;
            }
        }
        requireBoundary(selected === TILE_COUNT, "대한민국 좌표 타일 개수 오류");
        requireBoundary(bytes.length >= 128 && magic(bytes, "SRGKR004"),
            "대한민국 경계 데이터 버전 오류");
        const data = new DataView(bytes.buffer);
        const tileCount = data.getUint32(12, true);
        const partCount = data.getUint32(16, true);
        const pageCount = data.getUint32(20, true);
        const polygons = data.getUint32(24, true);
        const rings = data.getUint32(28, true);
        const vertices = data.getUint32(32, true);
        const columns = data.getUint32(36, true);
        const rows = data.getUint32(40, true);
        const scale = data.getUint32(44, true);
        requireBoundary(data.getUint32(8, true) === 128
            && tileCount > 0 && tileCount <= 262144 && tileCount === columns * rows
            && partCount > 0 && partCount <= 4096 && pageCount > 0 && pageCount <= 1000000
            && polygons > 0 && rings >= polygons && vertices >= rings * 4 && vertices <= 20000000
            && [1, 2, 4, 8, 16, 32, 64, 128, 256].includes(scale)
            && bytes.length === 128 + tileCount * 12 + partCount * 48
            && hexBytes(bytes.subarray(96, 128)) === SOURCE_SHA256,
        "대한민국 경계 루트 인덱스 오류");
        const gridWest = data.getFloat64(48, true);
        const gridSouth = data.getFloat64(56, true);
        const bounds = [64, 72, 80, 88].map(offset => data.getFloat64(offset, true));
        const [west, south, east, north] = bounds;
        requireBoundary(bounds.every(Number.isFinite)
            && west >= 124 && east <= 132 && west < east
            && south >= 32 && north <= 40 && south < north
            && gridWest === Math.floor(west * scale) / scale
            && gridSouth === Math.floor(south * scale) / scale
            && columns === Math.ceil(east * scale) - Math.floor(west * scale)
            && rows === Math.ceil(north * scale) - Math.floor(south * scale),
        "대한민국 경계 격자 범위 오류");
        let nextPage = 0;
        for (let tile = 0; tile < tileCount; tile++) {
            const entry = 128 + tile * 12;
            const state = data.getUint32(entry, true);
            const first = data.getUint32(entry + 4, true);
            const count = data.getUint32(entry + 8, true);
            requireBoundary(state <= 2 && first === nextPage && (count > 0) === (state === 2)
                && first + count <= pageCount, "대한민국 경계 타일 인덱스 오류");
            nextPage += count;
        }
        requireBoundary(nextPage === pageCount, "대한민국 경계 타일 페이지 누락");
        const partOffset = 128 + tileCount * 12;
        nextPage = 0;
        for (let number = 0; number < partCount; number++) {
            const entry = partOffset + number * 48;
            const first = data.getUint32(entry, true);
            const count = data.getUint32(entry + 4, true);
            const size = data.getUint32(entry + 8, true);
            requireBoundary(first === nextPage && count > 0 && first + count <= pageCount
                && data.getUint32(entry + 12, true) === 0
                && size > 64 + count * 12 && size <= MAX_PART_BYTES,
            "대한민국 경계 조각 인덱스 오류");
            nextPage += count;
        }
        requireBoundary(nextPage === pageCount, "대한민국 경계 조각 페이지 누락");
        return {bytes, data, bounds, gridWest, gridSouth, scale, columns, rows, tileIndex,
            tileCount, partCount, partOffset, pageCount, polygons, rings, vertices};
    })().catch(error => {
        boundaryPromise = null;
        throw error;
    });
    return boundaryPromise;
}
function partNumber(boundary, page) {
    let lower = 0;
    let upper = boundary.partCount;
    while (lower < upper) {
        const middle = Math.floor((lower + upper) / 2);
        const entry = boundary.partOffset + middle * 48;
        const end = boundary.data.getUint32(entry, true) + boundary.data.getUint32(entry + 4, true);
        if (end <= page) lower = middle + 1;
        else upper = middle;
    }
    requireBoundary(lower < boundary.partCount, "대한민국 경계 페이지 범위 오류");
    return lower;
}
async function loadPart(boundary, number) {
    return cached(parts, number, async () => {
        const root = boundary.partOffset + number * 48;
        const expectedFirst = boundary.data.getUint32(root, true);
        const expectedCount = boundary.data.getUint32(root + 4, true);
        const expectedSize = boundary.data.getUint32(root + 8, true);
        const digest = hexBytes(boundary.bytes.subarray(root + 16, root + 48));
        const name = "part-" + String(number).padStart(3, "0") + ".bin";
        const bytes = await download(name, MAX_PART_BYTES, digest);
        requireBoundary(bytes.length === expectedSize && magic(bytes, "SRGKP004"),
            "대한민국 경계 조각 구조 오류");
        const data = new DataView(bytes.buffer);
        requireBoundary(data.getUint32(8, true) === expectedFirst
            && data.getUint32(12, true) === expectedCount
            && data.getUint32(16, true) === expectedSize
            && data.getUint32(20, true) === 0
            && hexBytes(bytes.subarray(24, 56)) === SOURCE_SHA256
            && data.getBigUint64(56, true) === 0n,
        "대한민국 경계 조각 버전 혼합 또는 헤더 오류");
        let offset = 64 + expectedCount * 12;
        for (let local = 0; local < expectedCount; local++) {
            const entry = 64 + local * 12;
            const start = data.getUint32(entry, true);
            const length = data.getUint32(entry + 4, true);
            const rawLength = data.getUint32(entry + 8, true);
            requireBoundary(start === offset && length > 0 && start + length <= bytes.length
                && rawLength >= 32 && rawLength <= MAX_PAGE_BYTES,
            "대한민국 경계 압축 페이지 인덱스 오류");
            offset += length;
        }
        requireBoundary(offset === bytes.length, "대한민국 경계 조각 끝 위치 오류");
        return {bytes, data, first: expectedFirst};
    }, part => part.bytes.length);
}
function tilePosition(boundary, point) {
    const [latitude, longitude] = point;
    const [west, south, east, north] = boundary.bounds;
    requireBoundary(Number.isFinite(latitude) && Number.isFinite(longitude)
        && longitude >= west && longitude <= east && latitude >= south && latitude <= north,
    "대한민국 경계 판정 좌표 범위 오류");
    const column = Math.min(boundary.columns - 1,
        Math.floor((longitude - boundary.gridWest) * boundary.scale));
    const row = Math.min(boundary.rows - 1,
        Math.floor((latitude - boundary.gridSouth) * boundary.scale));
    requireBoundary(column >= 0 && row >= 0, "대한민국 경계 타일 소유 범위 오류");
    return row * boundary.columns + column;
}
async function loadPage(boundary, tile, pageNumber) {
    const page = await cached(pages, pageNumber, async () => {
        const part = await loadPart(boundary, partNumber(boundary, pageNumber));
        const entry = 64 + (pageNumber - part.first) * 12;
        const start = part.data.getUint32(entry, true);
        const length = part.data.getUint32(entry + 4, true);
        const rawLength = part.data.getUint32(entry + 8, true);
        const raw = inflate(part.bytes.subarray(start, start + length), rawLength);
        requireBoundary(raw.length === rawLength && magic(raw, "SRGKB004"),
            "대한민국 경계 압축 페이지 길이 또는 버전 오류");
        const data = new DataView(raw.buffer);
        const owner = data.getUint32(8, true);
        const kind = data.getUint32(16, true);
        const count = data.getUint32(20, true);
        const fieldA = data.getUint32(24, true);
        const fieldB = data.getUint32(28, true);
        requireBoundary(owner === tile && data.getUint32(12, true) === pageNumber,
            "대한민국 경계 페이지 식별자 오류");
        if (kind === 0) {
            requireBoundary(count <= 4092 && raw.length === 32 + count * 8
                && fieldA <= 1 && fieldB === 0, "대한민국 경계 패리티 페이지 오류");
            const row = Math.floor(tile / boundary.columns);
            const south = boundary.gridSouth + row / boundary.scale;
            const north = south + 1 / boundary.scale;
            let previous = south;
            for (let index = 0; index < count; index++) {
                const value = data.getFloat64(32 + index * 8, true);
                requireBoundary(Number.isFinite(value) && value > previous && value <= north,
                    "대한민국 경계 패리티 이벤트 범위 오류");
                previous = value;
            }
            return {tile, kind, count, base: fieldA, data};
        }
        requireBoundary(kind === 1 && count > 0 && count <= 1024
            && fieldA < boundary.rings && fieldB < boundary.polygons
            && raw.length === 40 + (count + 1) * 16, "대한민국 경계 링 페이지 오류");
        const first = data.getUint32(32, true);
        const hole = data.getUint32(36, true);
        requireBoundary(hole <= 1 && first + count < boundary.vertices,
            "대한민국 경계 링 정점 인덱스 오류");
        const bytes = new Uint8Array(raw.length);
        bytes.set(raw.subarray(0, 40));
        const words = (count + 1) * 2;
        for (let word = 0; word < words; word++) {
            for (let lane = 0; lane < 8; lane++) {
                bytes[40 + word * 8 + lane] = raw[40 + lane * words + word];
            }
        }
        const coordinates = new DataView(bytes.buffer);
        const [west, south, east, north] = boundary.bounds;
        for (let index = 0; index <= count; index++) {
            const longitude = coordinates.getFloat64(40 + index * 16, true);
            const latitude = coordinates.getFloat64(48 + index * 16, true);
            requireBoundary(Number.isFinite(longitude) && Number.isFinite(latitude)
                && longitude >= west && longitude <= east && latitude >= south && latitude <= north,
            "대한민국 경계 정점 범위 오류");
        }
        return {tile, kind, count, first, hole, ring: fieldA, polygon: fieldB, data: coordinates};
    }, decoded => decoded.data.byteLength);
    requireBoundary(page.tile === tile, "대한민국 경계 캐시 타일 식별자 오류");
    return page;
}
async function* tilePages(boundary, tile) {
    const entry = 128 + tile * 12;
    const first = boundary.data.getUint32(entry + 4, true);
    const count = boundary.data.getUint32(entry + 8, true);
    let geometry = false;
    let previousEvent = -Infinity;
    let previousRing = -1;
    let previousEnd = 0;
    let previousOwner = -1;
    let previousHole = -1;
    for (let offset = 0; offset < count; offset++) {
        const page = await loadPage(boundary, tile, first + offset);
        if (page.kind === 0) {
            requireBoundary(!geometry && (offset === 0 || page.base === 0),
                "대한민국 경계 패리티 페이지 순서 오류");
            if (page.count) {
                requireBoundary(page.data.getFloat64(32, true) > previousEvent,
                    "대한민국 경계 패리티 정렬 오류");
                previousEvent = page.data.getFloat64(32 + (page.count - 1) * 8, true);
            }
        } else {
            requireBoundary(offset > 0 && page.ring >= previousRing,
                "대한민국 경계 링 페이지 순서 오류");
            if (page.ring === previousRing) {
                requireBoundary(page.first >= previousEnd && page.polygon === previousOwner
                    && page.hole === previousHole, "대한민국 경계 링 메타데이터 불일치");
            }
            geometry = true;
            previousRing = page.ring;
            previousEnd = page.first + page.count;
            previousOwner = page.polygon;
            previousHole = page.hole;
        }
        yield page;
    }
    requireBoundary(geometry, "대한민국 경계 혼합 타일의 링 누락");
}
async function covers(boundary, point) {
    const [latitude, longitude] = point;
    const [west, south, east, north] = boundary.bounds;
    requireBoundary(Number.isFinite(latitude) && Number.isFinite(longitude),
        "대한민국 경계 판정 좌표 범위 오류");
    if (longitude < west || longitude > east || latitude < south || latitude > north) return false;
    const tile = tilePosition(boundary, point);
    const state = boundary.data.getUint32(128 + tile * 12, true);
    if (state !== 2) return state === 1;
    const location = [longitude, latitude];
    let parity = false;
    let boundaryHit = false;
    for await (const page of tilePages(boundary, tile)) {
        if (page.kind === 0) {
            let lower = 0;
            let upper = page.count;
            while (lower < upper) {
                const middle = Math.floor((lower + upper) / 2);
                if (page.data.getFloat64(32 + middle * 8, true) <= latitude) lower = middle + 1;
                else upper = middle;
            }
            parity = parity !== Boolean(page.base ^ (lower & 1));
            continue;
        }
        let start = [page.data.getFloat64(40, true), page.data.getFloat64(48, true)];
        for (let index = 1; index <= page.count; index++) {
            const end = [page.data.getFloat64(40 + index * 16, true),
                page.data.getFloat64(48 + index * 16, true)];
            if (latitude >= Math.min(start[1], end[1]) && latitude <= Math.max(start[1], end[1])) {
                const side = orientation(start, end, location);
                if (side === 0 && longitude >= Math.min(start[0], end[0])
                    && longitude <= Math.max(start[0], end[0])) boundaryHit = true;
                if ((start[1] > latitude) !== (end[1] > latitude)) {
                    const crossing = end[1] > start[1] ? side > 0 : side < 0;
                    parity = parity !== crossing;
                }
            }
            start = end;
        }
    }
    return boundaryHit || parity;
}
function remapTile(value, boundary) {
    const selector = Number(value >> 46n);
    if (selector >= TILE_COUNT * 3) return null;
    const tile = boundary.tileIndex[Math.floor(selector / 3)];
    const vertical = Math.floor(tile / 512) * 8388608 + Number((value >> 23n) & 0x7fffffn);
    const horizontal = (tile % 512) * 8388608 + Number(value & 0x7fffffn);
    return (BigInt(vertical) << 32n) | BigInt(horizontal);
}
export async function* sampleKorea(value) {
    requireBoundary(typeof value === "bigint" && value >= 0n && value <= MAX_U64,
        "64비트 난수 범위를 벗어났습니다.");
    const boundary = await loadBoundary();
    let candidateValue = value;
    for (let supplementalReads = 0; supplementalReads <= SUPPLEMENT_LIMIT; supplementalReads++) {
        const mapped = remapTile(candidateValue, boundary);
        if (mapped !== null) {
            const point = candidate(mapped, boundary.bounds);
            if (await covers(boundary, point)) return point;
        }
        requireBoundary(supplementalReads < SUPPLEMENT_LIMIT,
            "대한민국 좌표 보완 난수 시도 횟수 초과");
        const supplement = yield;
        requireBoundary(supplement?.bitsRemaining === 64 && typeof supplement.value === "bigint"
            && supplement.value >= 0n && supplement.value <= MAX_U64,
        "대한민국 좌표에는 새로운 전체 64비트 보완 난수가 필요합니다.");
        candidateValue = supplement.value;
    }
}
export function prepareKorea(value) {
    requestedPreparation = value;
    if (!preparation) preparation = (async () => {
        let current;
        do {
            current = requestedPreparation;
            const boundary = await loadBoundary();
            if (typeof current === "bigint" && current >= 0n && current <= MAX_U64) {
                const mapped = remapTile(current, boundary);
                if (mapped !== null) {
                    const tile = tilePosition(boundary, candidate(mapped, boundary.bounds));
                    if (boundary.data.getUint32(128 + tile * 12, true) === 2) {
                        for await (const _page of tilePages(boundary, tile)) {
                            if (current !== requestedPreparation) break;
                        }
                    }
                }
            }
        } while (current !== requestedPreparation);
    })().finally(() => {
        preparation = null;
    });
    return preparation;
}
