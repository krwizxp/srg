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
function cumulativeArea(latitude, withDerivative = false) {
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
    const area = sine * polynomial;
    if (!withDerivative) return area;
    const cosine = 0.8090169943749475 * cosineDelta - 0.5877852522924731 * sineDelta;
    const denominator = 1 - squareEccentricSine;
    return [area, DEGREES * cosine / (denominator * denominator)];
}
function candidate(value, boundary) {
    const bounds = boundary.bounds;
    if (!Array.isArray(value) || value.length !== 2
        || value.some(word => !Number.isInteger(word) || word < 0 || word > 0xffffffff)) {
        throw new Error("대한민국 등면적 좌표 입력 오류");
    }
    const [west, south, east, north] = bounds;
    if (!(bounds.length === 4 && bounds.every(Number.isFinite)
        && west >= -180 && east <= 180 && west < east
        && south >= 32 && north <= 40 && south < north)) {
        throw new Error("대한민국 등면적 좌표 범위 오류");
    }
    const vertical = (value[0] + 0.5) / 4294967296;
    const horizontal = (value[1] + 0.5) / 4294967296;
    const southArea = boundary.southArea;
    const target = southArea + vertical * (boundary.northArea - southArea);
    let latitude = south + (north - south) * vertical;
    for (let iteration = 0; iteration < 3; iteration++) {
        const [area, derivative] = cumulativeArea(latitude, true);
        latitude -= (area - target) / derivative;
    }
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
function unpackBootstrap(compact) {
    requireBoundary(compact.length >= 32904 && compact.length <= 557192
        && magic(compact, "SRGKC001"), "대한민국 경계 압축 인덱스 버전 오류");
    const header = new DataView(compact.buffer, compact.byteOffset + 8, 128);
    const tileCount = header.getUint32(12, true);
    const partCount = header.getUint32(16, true);
    const pageCount = header.getUint32(20, true);
    requireBoundary(tileCount > 0 && tileCount <= 262144
        && partCount > 0 && partCount <= 4096 && pageCount > 0 && pageCount <= 1000000,
    "대한민국 경계 압축 인덱스 범위 오류");
    const stateBytes = Math.ceil(tileCount / 4);
    requireBoundary(compact.length >= 136 + stateBytes + partCount * 48 + 32768,
        "대한민국 경계 압축 인덱스 길이 오류");
    let mixed = 0;
    for (let tile = 0; tile < tileCount; tile++) {
        const state = (compact[136 + (tile >>> 2)] >>> ((tile & 3) * 2)) & 3;
        requireBoundary(state <= 2, "대한민국 경계 타일 인덱스 오류");
        if (state === 2) mixed++;
    }
    requireBoundary((tileCount % 4 === 0
        || compact[135 + stateBytes] >>> ((tileCount % 4) * 2) === 0)
        && compact.length === 136 + stateBytes + mixed + partCount * 48 + 32768,
    "대한민국 경계 압축 인덱스 길이 오류");
    const bytes = new Uint8Array(128 + tileCount * 12 + partCount * 48);
    const data = new DataView(bytes.buffer);
    bytes.set(compact.subarray(8, 136));
    let cursor = 136 + stateBytes;
    let nextPage = 0;
    for (let tile = 0; tile < tileCount; tile++) {
        const state = (compact[136 + (tile >>> 2)] >>> ((tile & 3) * 2)) & 3;
        const count = state === 2 ? compact[cursor++] : 0;
        requireBoundary((count > 0) === (state === 2) && nextPage + count <= pageCount,
            "대한민국 경계 타일 인덱스 오류");
        const entry = 128 + tile * 12;
        data.setUint32(entry, state, true);
        data.setUint32(entry + 4, nextPage, true);
        data.setUint32(entry + 8, count, true);
        nextPage += count;
    }
    requireBoundary(nextPage === pageCount, "대한민국 경계 타일 페이지 누락");
    bytes.set(compact.subarray(cursor, cursor + partCount * 48), 128 + tileCount * 12);
    const tileBytes = compact.slice(cursor + partCount * 48);
    return [bytes, tileBytes];
}
async function verifyBootstrap(compact) {
    requireBoundary(typeof globalThis.crypto?.subtle?.digest === "function",
        "대한민국 경계 무결성 검증을 지원하는 HTTPS 환경이 필요합니다.");
    const unpacked = unpackBootstrap(compact);
    const digests = await Promise.all(unpacked.map(bytes => crypto.subtle.digest("SHA-256", bytes)));
    requireBoundary(hexBytes(new Uint8Array(digests[0])) === INDEX_SHA256
        && hexBytes(new Uint8Array(digests[1])) === TILE_SHA256,
    "대한민국 경계 SHA-256 불일치");
    return unpacked;
}
function decodeAtlas(bytes, boundary) {
    requireBoundary(bytes.length >= 96 && magic(bytes, "SRGKA001"),
        "대한민국 세부 격자 데이터 버전 오류");
    const data = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const subdivision = data.getUint32(12, true);
    const mixedCount = data.getUint32(16, true);
    const stateBytes = data.getUint32(88, true);
    requireBoundary(data.getUint32(8, true) === 96
        && subdivision === 8
        && mixedCount > 0 && mixedCount <= boundary.tileCount
        && boundary.tileCount <= 65536
        && data.getUint32(20, true) === boundary.tileCount
        && stateBytes === mixedCount * subdivision * subdivision / 4
        && bytes.length === 96 + stateBytes && data.getUint32(92, true) === 0
        && hexBytes(bytes.subarray(24, 56)) === INDEX_SHA256
        && hexBytes(bytes.subarray(56, 88)) === SOURCE_SHA256,
    "대한민국 세부 격자 인덱스 오류");
    const states = bytes.subarray(96);
    for (const value of states) {
        requireBoundary((value & (value >>> 1) & 0x55) === 0,
            "대한민국 세부 격자 상태 오류");
    }
    const mixedOrdinal = new Uint16Array(boundary.tileCount).fill(65535);
    let ordinal = 0;
    for (let tile = 0; tile < boundary.tileCount; tile++) {
        if (boundary.data.getUint32(128 + tile * 12, true) === 2) {
            requireBoundary(ordinal < mixedCount, "대한민국 세부 격자 개수 오류");
            mixedOrdinal[tile] = ordinal++;
        }
    }
    requireBoundary(ordinal === mixedCount, "대한민국 세부 격자 개수 오류");
    return {subdivision, states, mixedOrdinal};
}
function atlasState(boundary, point, tile) {
    const {subdivision, states, mixedOrdinal} = boundary.atlas;
    const lower = mixedOrdinal[tile];
    requireBoundary(lower !== 65535, "대한민국 세부 격자 소유 범위 오류");
    const row = Math.floor(tile / boundary.columns);
    const column = tile % boundary.columns;
    const fineColumn = Math.min(subdivision - 1,
        Math.floor((point[1] - boundary.gridWest) * boundary.scale * subdivision)
        - column * subdivision);
    const fineRow = Math.min(subdivision - 1,
        Math.floor((point[0] - boundary.gridSouth) * boundary.scale * subdivision)
        - row * subdivision);
    requireBoundary(fineColumn >= 0 && fineRow >= 0, "대한민국 세부 격자 좌표 범위 오류");
    const index = (lower * subdivision + fineRow) * subdivision + fineColumn;
    return (states[index >>> 2] >>> ((index & 3) * 2)) & 3;
}

const ATLAS_SHA256 = "a55dcf2e37a4500d6c30de3f5dad42131758415bcfafac79e88d76fb88b0e84d";
function requestAtlas(boundary) {
    if (boundary.atlasRequested || ++boundary.mixedChecks < 12) return;
    boundary.atlasRequested = true;
    setTimeout(async () => {
        try {
            const bytes = await download("atlas-8.bin", 41328, ATLAS_SHA256);
            boundary.atlas = decodeAtlas(bytes, boundary);
        } catch {
            boundary.atlasUnavailable = true;
        }
    }, 0);
}
async function loadBoundary() {
    if (!boundaryPromise) boundaryPromise = (async () => {
        const compact = Uint8Array.fromBase64("U1JHS0MwMDFTUkdLUjAwNIAAAAAZoQAAYAAAAARjAACFNQAAkTUAAOFtTwDpAAAAsQAAACAAAAAAAAAAACZfQAAAAAAAjEBAVE9MnQUnX0C4omqsYY5AQEAq7Nbte2BA5fjBBo5OQ0BQYPiCQeLYC4fwUrS4MpbvWc1/TuBmqTMLwupYC8VUKAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAKgACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgKqqqioAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIBWVVWpCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAVlVVVaUKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFpVVVVVpQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgVVVVVVUKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGpVVVVVpQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgVVVVVVUCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGpVVVVVKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgqlVVVakCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACqVlWpCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKqqqgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgCoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACoAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKAIAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKoKAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACACgAAAAAAAIACAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoAAAAAAAAAAAAAAAAAAAAACAqgAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKACAAAAAAAAAAAAAAAAAAAAAAgCIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAgAAAAAAAAAAAACoKgAAAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIACAAAAqKoIqAAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgAAAKiqoqgCCAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoAAAiAIIAACoqoqCiiKqAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAICqqiqAKKAKCAgKgCqAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoKiqAKKKCgAAIKKgKqIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAICoKgAoqCqqqKiqgqoIAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoIoKCKCqmqqqqoKgAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKgqqiqAWlqqqqqCCAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAIAAACAqmipCmqpqqqqCioACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIACAAAAAKBqVaqqpaqqqqqqCqgCAIIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgAAAACAqlaVqFalqqqqpaigKgAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAAAAAKCqVYpWVWqVgqqqqgoAqACAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAgCgqlqpWlWlVQqoqqWqAKoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAgKqqqlpVlValYGqVKoCqAgAACigKACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIKqqqqpWVVValYJaVaqqqgoAACigAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAKCKqqpqVVVVaZWKalVVqqoqAAACqgCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgCgAAKKqqolVVVVVVqoJaVaoqmgAIAKgqigIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKoAoAACCqiqpVVVVVVVUqoFqqqqoCCAqoKgCoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKAAqAAAAqqKqpqpVVVVVlaqqKqiiCqAqKAqiqAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAqAIAAKiqqqiqVVVVVVWVqqqqqqqqqgAgqKoqAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIACAACgqqqKqlZVVVVVVammqqYpqpYCiqiqqgoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgqqqCqpaVVVVVVVVlapalWhZCqigqloqAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgqoqqWlVVVVVVVZWqWlpqqaqqoqqqqQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgKqqqmpVVVVVVVVVpZqqqqmqoIKqlpUCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABqqqpqVVVVVVVVVVVpqqqqqqqqqqpqKgACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgKqqqlVVVVVVVVVVVamqaqqqqmqqaikAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCIqqpWVVVVVVVVVVVVpaqqWqWqqaqmAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgKqqWlVVVVVVVVVVVVVZqWpVVamqqooAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqoqqpVVVVVVVVVVVVVVZWqVVVpKaiqCggAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqqqqqWlVVVVVVVVVVVVVVpVZVpSqqqqqqKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqKpqqmpVVVVVVVVVVVVVVVVZVVWpqqqqqqoCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACqopqqVVVVVVVVVVVVVVVVVVVVlaqqqqmpCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgAKqqqlVVVVVVVVVVVVVVVVVVVVVVqlZVlaoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACqhVVVVVVVVVVVVVVVVVVVVVVWlVVVWVKgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKgKqqVlVVVVVVVVVVVVVVVVVVVVWlVlVVVZUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIACoFVVVVVVVVVVVVVVVVVVVVVVVVlVVVVVCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKohaVVVVVVVVVVVVVVVVVVVVVVVVVVVVVSkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAICoqFVVVVVVVVVVVVVVVVVVVVVVVVVVVVWVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAKAIpWVVVVVVVVVVVVVVVVVVVVVVVVVVVVVSoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKggAqFVVVVVVVVVVVVVVVVVVVVVVVVVVVVWVAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAiAKBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAWlVVVVVVVVVVVVVVVVVVVVVVVVVVVVWlAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGhVVVVVVVVVVVVVVVVVVVVVVVVVVVVVpQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAVlVVVVVVVVVVVVVVVVVVVVVVVVVVVZWqAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKhaVVVVVVVVVVVVVVVVVVVVVVVVVVVVqQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCqqlZVVVVVVVVVVVVVVVVVVVVVVVVVVZUJAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAICKgKpaVVVVVVVVVVVVVVVVVVVVVVVVVVVVJQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqopWVVVVVVVVVVVVVVVVVVVVVVVVVVVVVZUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKKAIWlVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKAAAKhaVVVVVVVVVVVVVVVVVVVVVVVVVVVVVSkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAalVVVVVVVVVVVVVVVVVVVVVVVVVVVVWVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgKpWVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqWlVVVVVVVVVVVVVVVVVVVVVVVVVVVVUJAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgKqhaVVVVVVVVVVVVVVVVVVVVVVVVVVVVpQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgKqCalVVVVVVVVVVVVVVVVVVVVVVVVVVVVUCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIICqCmBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAACCqVVVVVVVVVVVVVVVVVVVVVVVVVVVVVaUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAalVVVVVVVVVVVVVVVVVVVVVVVVVVVVWVAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKpVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgCqqlpVVVVVVVVVVVVVVVVVVVVVVVVVVaWlAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAKhqVVVVVVVVVVVVVVVVVVVVVVVVVVWlagIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAICqVlVVVVVVVVVVVVVVVVVVVVVVVVVVpYIKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAooFZVVVVVVVVVVVVVVVVVVVVVVVVVVVUqKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAgAAAKBaVVVVVVVVVVVVVVVVVVVVVVVVVVVVpQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoAAACoWlVVVVVVVVVVVVVVVVVVVVVVVVVVVaUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoFVVVVVVVVVVVVVVVVVVVVVVVVVVVVWVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKoAAABaVVVVVVVVVVVVVVVVVVVVVVVVVVVVlQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAICqAigAaFVVVVVVVVVVVVVVVVVVVVVVVVVVVVUJAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAiqKohVVVVVVVVVVVVVVVVVVVVVVVVVVVVVKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqCqgVlVVVVVVVVVVVVVVVVVVVVVVVVVVVaUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgAlpVVVVVVVVVVVVVVVVVVVVVVVVVVVVVAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAiKpoVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACqalVVVVVVVVVVVVVVVVVVVVVVVVVVVVWlAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIqKpVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgKiJalVVVVVVVVVVVVVVVVVVVVVVVVVVVVUpAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgKaZWVVVVVVVVVVVVVVVVVVVVVVVVVVVVpQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgAKaqVlVVVVVVVVVVVVVVVVVVVVVVVVVVVVUCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACoqFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACoqGJVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKAAAACAgKKqVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUlAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAgIIICqAWVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVlQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgKACqqlZVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUqAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoKpWVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVpQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgACqWlVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVZUCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIBWVlVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVSkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaqpVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUpAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKilVlVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVJQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgqqpVVVWqVVVVVVVVVVVVVVVVVVVVVVVVVZUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAKqqVlVVqlZVVVVVVVVVVVVVVVVVVVVVVVVVAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAIqFpVVapVVVVVVVVVVVVVVVVVVVVVVVVVVQkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAigAAAKCaVamqVVVVVVVVVVVVVVVVVVVVVVVVVVUlAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKAAAqpqqqlVVVVVVVVVVVVVVVVVVVVVVVVVVlQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKgAAACqCqhVVVVVVVVVVVVVVVVVVVVVVVVVVVUCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgKgAAqAqgVlVVVVVVVVVVVVVVVVVVVVVVVVVVCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKgAAIgoqlZVVVVVVVVVVVVVVVVVVVVVVVVVVQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoAKAKAKiqVVVVVVVVVVVVVVVVVVVVVVVVVVUJAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqCioqgCgqlZVVVVVVVVVVVVVVVVVVVVVVVVVKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKigqoCiqFpVVVVVVVVVVVVVVVVVVVVVVVVVVaUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgKqgqqpqVVVVVVVVVVVVVVVVVVVVVVVVVVWVAgAAAAAAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAAAAAAqoIqqqmpqVVVVVVVVVVVVVVVVVVVVVVVVVQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKAAAKKqqWlVVVVVVVVVVVVVVVVVVVVVVVVUKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgCgalVVVVVVVVVVVVVVVVVVVVVVVVVVCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgiKpWVVVVVVVVVVVVVVVVVVVVVVVVVQkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoAKYqlVVVVVVVVVVVVVVVVVVVVVVVVUqAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAqqKVWVVVVVVVVVVVVVVVVVVVVVVVVCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAmqJWVVVVVVVVVVVVVVVVVVVVVVVVVSkAAAAAAAAAAAAAAACgAgAAAAAAAAAAAAAAAAAAAAAAAAAAAKqqWlVVVVVVVVVVVVVVVVVVVVVVVVUqAAAAAAAAAAAAAAAAoCkAAAAAAAAAAAAAAAAAAAAAAAAAAACqqmpVVVVVVVVVVVVVVVVVVVVVVVVVKQAAAAAAAAAAAAAAAICWAgAAAAAAAAAAAAAAAAAAAAAAAACAqqpqVVVVVVVVVVVVVVVVVVVVVVVVVSUAAAAAAAAAAAAAAAAAqAoAAAAAAAAAAAAAAAAAAAAAAAAAAACoalVVVVVVVVVVVVVVVVVVVVVVVVWlAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAIAAgAAqqpVVVVVVVVVVVVVVVVVVVVVVVVVpQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAKAAAAKqqlVVVVVVVVVVVVVVVVVVVVVVVVVZUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgCAoCqqllZVVVVVVVVVVVVVVVVVVVVVVVVVAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAICiqlZaVVVVVVVVVVVVVVVVVVVVVVVVVQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgKpWWVVVVVVVVVVVVVVVVVVVVVVVVVUKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoAAAAAAAAAAAAAACqWqWqVVVVVVVVVVVVVVVVVVVVVVWVCgAAAAAAAAAAAAAAAAAAAAAAAAAAAACAigAAAAAAAAAAAAAAqKqpAlZVVVVVVVVVVVVVVVVVVVVVlQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKgAAAAAAAAAAAAAAIACKgBoVVVVVVVVVVVVVVVVVVVVVZUCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAgAAAAAAAAAAAAAAAAAAoFVVVVVVVVVVVVVVVVVVVVWVAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIBVVVVVVVVVVVVVVVVVVVVVVQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAKAAAAAAAAAAAAAAAAAAAAWlVVVVVVVVVVVVVVVVVVVVUKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACApgAAAAAAAAAAAAAAAAAAAKhVVVVVVVVVVVVVVVVVVVVVCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKoCAAAAAAAAAAAAAAAAAAAAqlVVVVVVVVVVVVVVVVVVVQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABWVVVVVVVVVVVVVVVVVVUKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaFVVVVVVVVVVVVVVVVVVCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIBVVVVVVVVVVVVVVVVVVQkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAalVVVVVVVVVVVVVVVVUpAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIBaVVVVVVVVVVVVVVVVKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaFVVVVVVVVVVVVVVVSUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKBWVVVVVVVVVVVVVVWlAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaFVVVVVVVVVVVVVVlQIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIBaVVVVVVVVVVVVVZUCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoFZWpVVVaZVaVVVVAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACqqqqqqqqqqlpVVQoAAAAA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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+AEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP4H4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+B+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/gfgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgP////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAID/////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACA/////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD/////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg/////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P//////PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz//////z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8//////8/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P///////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz///////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P///////wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD///////8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg////////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P///////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAID///////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACA////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgP///////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+//////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v//////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP7//////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4//////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+P//////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPj///////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P///////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD///////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAID///8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACA////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgP///wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8D8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwPwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgBwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOAfAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgHwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4B8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOB/AMABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAfwDAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4H8AwAEAHgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOB3AAAAAB4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgdwAAAAAeAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/n8AAAAAHgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP5/AAAAAB4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+fwAAAAAeAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/n8AAAAAHgAA8AEAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOAfAAAAAAAAAPABABwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgHwAAAAAAAADwAQAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8AAAAAAAAAAAA4B8AAAAAAAAA/D8AHwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAfAAAAAAAAAAAAAAAAAAAAAAAAPw/AB8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHwAAAAAAAAAAAAAAAAAAAAAAAD8PwAfAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIB/AAAAAAAAAAAAAAAAAAAAAAAA/D8AHwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAfwAAAAAAAAAAAAAAAAAAAAAAAHA8AAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgH8AAAAAAAAAAAAAAAAAAAAAAABwPAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIB/AAAAAAAAAAAAAADg/wcAAAAAcDw4BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAHwAAAAAAAAAAAAAA4P8HAAAAAAAAOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgB8AAAAAAAAAAAAAAOD/BwAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAfAAAAAAAA+AAAAAD4/3fgHwAAAAD4AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPgAAAAA+P934B8AAAAA4AEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4AAAAAPj/d+AfAAAAAOABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+AAAAAD///f7H/AAAAfgAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPgAAAAA///3+x/wAAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4AAAAAP//9/sf8AAABwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8AADc/3AAAMD///f73/f+HwcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAfAAA3A9wAADA///349/3/h8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHwAANwPcAAAwP//9+Pf9/4fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8AAD8////APf///fj3/f+f4DjAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAA8P///wD3gf/Bg8MH4H+A4wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAAPD///8A94H/wYPDB+B/gOMBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABwAADw////APe//8GDw/f+/7/7AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAL7//wD3v38AAAD3Pvy/+wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC+//8A979/AAAA9z78v/sBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v//AP//f//78/8//L/7AQB4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAO7/BwAf/3//+/P/P/y/AwAAeAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADu/wcAH/9///vz/z/8vwMAAHgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7v//4B////////+//78DAAB4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOB//+AB////////j8c/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgf//gAf///////4/HPwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/n///wH////////v3z84AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP5///8B////////4xwAOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+f///Af///////+McADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgA8AAAAA/n///x/////////jHwA+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAPAAAAAPh///8f////////4x8ADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACADwAAAAD4f///H////////+MfAA4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4A8AAAAA+P///////////////4H/AYDDAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOADAAAAAID///////////////+B/wGAwwEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAwAAAACA////////////////gf8BgMMBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AMAAAAAgP///////////////4f/B4DPAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOADAAAAAID////f//////////eH/weADwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAwAAAACA////3//////////3h/8HgA8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AMAAAAAgP//////////////////B+APADgAAAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIADAAAAAADw//9////////x////PwDgDwA4AAAABwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAwAAAAAA8P//f///////8f///z8A4A8AOAAAAAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAMAAH4AAPz///////////H/////Af8PADgAAAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB+AAD8//////////+B/////wH/AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAfgAA/P//////////gf////8B/wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH4ADvz//////////4f/////8f8DAAB84PsBAA4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA78//////////+H////D/D/AwAAfOD7AQAOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAO/P//////////h////w/w/wMAAHzg+wEADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABw/v///////////4f///////8DAAB84PtxAA4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcP7///////////+H////////AwAAfOADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHD+////////////h////////wMAAHzgA3AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAD8//////////////f///////8DAOD8/wPwAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABwAA/O/////////////3////////AwDgwP8A8AEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAPzv////////////9////////wMA4MD/APABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOB/AAD87/////////////////////8DcODA/+/3AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgfwAAwO////v/////////+P///3//A3AAwP/v9wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4H8AAMDv///7//////////j///9//wNwAMD/7/cBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAB/h/AADA///////////////7//////8DfD7w/+/3PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAf4HwAAwPv//f//////////g///////Axw+8P8DwD8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAH+B8AAMD7//3//////////4P//////wMcPvD/A8A/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAB/gfAADA//////////////////////8D/D/8/+/3PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAfgHwAAAP7f////////////////43//A/w/fD/u9z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAH4B8AAAD+3////////////////+N//wP8P3w/7vc/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAB+B/AAAA/v///////////////////////z98P/7//wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAfgfwAAAP7///v///////////////////8/ADz+//8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH4H8AAAD+///7////////////////////PwA8/v//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB+B/AAAA/v///////////////////////z/wvf///w8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHwAAAP7//+///////////////////v8/8L3///8PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8AABz+///v//////////////////7/P/C/////DwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAc/v//4//////////////////+/z/wh////w8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHP7//+P//////////////////v8/8If///8PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABz+///////////////////////////n////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P///v//////////////////////5////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD///7//////////////////////+f///9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg///////////////////////////n////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P///////////////////////z/84f///w8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD///////////////////////8//OH///8PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg////////////////////////////////fwAOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P///////////////////////////////38ADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD///////////////////////////////9/AA4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg////////////////////////////////fwA+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz//////////////////////////////w8AOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8//////////////////////////////8PADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg/P//////////////////////////////DwA4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4HD//////////////////////////////w8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOBw//////////////////////////////8PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg8P//////////////////////////////fwc4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMD//////////////////////////////38HOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADA//////////////////////////////9/BzgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgx////////////////////////////////x84AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4Af///////////////////////////8f//8fOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOAH////////////////////////////H///HzgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD49///////////////////////////////////fgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+Pf////////////////////////////3/////34AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPj3////////////////////////////9/////9+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4/////////////////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+P////////////////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPj/////////////////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4/////////////////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgP/n//////////////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAID/5///////////////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC+/////////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPsD///////////////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD7A////////////////////////////////////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+wP////////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgA///////////////////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AP//////////////////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+AP////////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPgD/////////////////////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4A/////////////////////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+wP////////////////////////////////////8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAH4P//////////////////////////////////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAB+D//////////////////////////////////wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwD/u//////////////////////////////////8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMA/7v//////////////////////////////////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAP+7//////////////////////////////////wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwL////////////////////////////////////8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC8P///////////////////////////////////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIB/vP///////////////////////////////////z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAfwD4/f////////////////////////////////8/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgH8A+P3/////////////////////////////////PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIB/PPj9//////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAfzwA8P//////////////////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgH88APD//////////////////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAID/PQDw//////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8D0A8P//////////////////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPA9APD//////////////////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwPQDw//////////////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8P//////////////////////////////////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPD//////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADw//////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwP//////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMD//////////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADA//////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPj/////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4/////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADc//////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA3P//////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANz//////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP7d//////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+Hfj/////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/h34/////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP5///////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+f///////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/n///////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/5///////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/4Hf//////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/+B3//////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/gd////////////////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8AAP///////////////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/AAD///////////////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPwAA////////////////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD//////////////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD///////////////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA////////////////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP///////////////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD///////////////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//7/////////////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/+/////////////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz//v////////////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8P/7/////////////////////////////////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/D/+/////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz//v////////////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8//j/////////////////////////////////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P/4/////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAP//+P////////////////////////////////8/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AD//wD/////////////////////////////////PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOAA//8A/////////////////////////////////z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMDhAP///v////////////////////////////////8/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAQAA4P7/////////////////////////////////PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAEAAOD+/////////////////////////////////z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMABAADg//////////////////////////////////8/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P//////////////////////////////////PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD//////////////////////////////////z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg//////////////////////////////////8/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P//////////////////////////////////PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD//////////////////////////////////z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgD4////////////////////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4A+P///////////////////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOAPj///////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgD4////////////////////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AAPj//////////////////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOAAD4//////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgDg////////////////////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P////////////////////////////////8f/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwO//////////////////////////////////f/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAP/v///////////////////////////////3/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAD/7///////////////////////////////9//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgBwAAwI//////////////////////////////////f/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AcAAACA/////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOAHAAAAgP////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgBwAAAPj/////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AcAAAD4/////////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOAHAAAA+P////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgBwAAAPj/////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4/////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+P////////////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/AcAAPj/////////////////////////////////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPwHAADg/////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8BwAA4P////////////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg/wf4AeD/////////////////////////////////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P8H+AHg/////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD/B/gB4P////////////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg/4f/f/j/////////////////////////////////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAeH/3+4/////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAHh/9/uP////////////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAB4f/f/j/////////////////////////////////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACA/x/4/////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgP8f+P////////////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAID/f/j/////////////////////////////////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH/g/////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADj/7/////////////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4/+//////////////////////////////////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOP/v/////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADj///////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA////////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP///////////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAP///////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAD////////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAA////////////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgO////////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADu//n/////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7v/5/////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAO7/+f////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg//n/////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P/5/////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/OD///////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPwA////////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8AP///////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/AD///////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAf///////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH///////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4O9///////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAODvf///////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg73///////////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8AAADA4e////////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/AAAAwIHv////////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP3jAA9y/7////////////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD94wAPcP+D///////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/eMAD3D/g////////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/jxA/z///////////////////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADA8QPw////////////////////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwPED8P///////////////////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMDxA/D///////////////////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADA////////////////////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwP///////////////////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4MD///////////////////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOAA////////////////////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAP///////////////////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AD///////////////////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA////////////////////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP///////////////////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD///////////////////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P//////////////////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz//////////////////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8//////////////////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P////////////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz/////////////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8/////////////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P////////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz/////////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8/////////////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P////////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABwAAPz/////////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAADg/////////////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAA4P////////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHwAAOD/////////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABwAADg4P///////////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAA4OD///////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwHcAAODg////////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMB3AAAA4P///////////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAdwAAAOD///////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwHeAHwDg////////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgB8AAP///////////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAfAAD///////////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgHwAA////////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4B8AAMD/D/z//////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOAfAADA/w/8//////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg/wEAwP8P/P//////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P8BAMD/A/z//////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD/AQDA/wP8//////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg/wEA8P/7////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPwBAHDv+////////////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAID/Afh/7/v///////////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAHwD4HwD4////////////////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgB8A+B8A+P///////////////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAODfh/9/APj///////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg34f/fwD4////////////////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4N+H/38A+P///////////////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPjf//9/v/////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4x/8PfL//////////////////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+Mf/D3y//////////////////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPjH/+////////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAwP/j////////////////////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAA4AAAAAAAAAAAAAAAAMD/4////////////////////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAOAAAAAAAAAAAAAAAADA//v///////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAwD/43/////////////////////////////////8/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMA/+N//////////////////////////////////PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAP/jf/////////////////////////////////z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAcAAL////////////////////////////////8/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAHAAC/////////////////////////////////PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADABzgAv////////////////////////////////z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AAD+//////////////////////////////8PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOMDv/v//////////////////////////////DwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADA7/7//////////////////////////////wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwO/+//////////////////////////////8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPDv/v//////////////////////////////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwD/j//////////////////////////////wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8A/4//////////////////////////////8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPA//v//////////////////////////////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwP/7/////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8D/+/////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz//v////////////////////////////8fAAAAAAAAAAAAAAAAAOAPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8//7/////////////////////////////HwAAAAAAAAAAAAAAAADgDwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P/+/////////////////////////////x8AAAAAAAAAAAAAAAAA4A8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz///////////////////////////////8fAAAAAAAAAAAAAAAAAPh/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8////////////////////////////////BwAAAAAAAAAAAAAAAAD4fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P///////////////////////////////wcAAAAAAAAAAAAAAAAA+H8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP////////////////////////////////8HAAAAAAAAAAAAAAAAAPj/AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/////////////////////////////////AQAAAAAAAAAAAAAAAAD4/wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/////////////////////////////////wEAAAAAAAAAAAAAAAAA+P8BAAAAAAAAAAAAAAAAAAAAAAAAAAAA+P////////////////////////////////8BAAAAAAAAAAAAAAAAAPj/AQAAAAAAAAAAAAAAAAAAAAAAAAAAAPj///////////////////////////////9/AAAAAAAAAAAAAAAAAADg/wEAAAAAAAAAAAAAAAAAAAAAAAAAAAD4////////////////////////////////fwAAAAAAAAAAAAAAAAAA4P8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMD//////////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADA//////////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+AGABwAA/P//////////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPgBgAcAAPz//////////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4AYAHAAD8//////////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/gGAB4A/////////////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP4BAACAP////////////////////////////////wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+AQAAgD////////////////////////////////8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/gEAHP//////////////////////////////////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH4AABz/7////////////////////////////////wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB+AAAc/+////////////////////////////////8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAfgAAHP//////////////////////////////////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC//////////////////////////////////wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv/////////////////////////////////8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAL//////////////////////////////////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACA/////////////////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAHwAAAAAAAAAAAAAAgP////////////////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgB8AAAAAAAAAAAAAAID///////////////////////////////8/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAfAAAAAAAAAAAAAACA////////////////////////////////PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+HwAAAAAAAAAAAAAAgP///////////////////////////////z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/h4AAAAAAAAAAAAAAID///8D//////////////////////////8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP4eAAAAAAAAAAAAAACA////A///////////////////////////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+HwAAAAAAAAAAAAAAgP///wP//////////////////////////wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+AMAAAAAAAAAAAAAAAA+8A8A/////////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPgDAAAAAAAAAAAAAAAAPvAPAP////////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4AwAAAAAAAAAAAAAAAD7wDwD/////////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AMAAAAAAAAAAAAAAAAAAAAA/////////////////////////x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOADAAAAAAAAAAAAAAAAAAAAAP////////////////////////8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAwAAAAAAAAAAAAAAAAAAAAD/////////////////////////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+AAAAAAAAAAAAAAAAAAAAAAD/////////////////////////BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/gAAAAAAAAAAAAAAAAAAAAAA/////////////////////////wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP8DAAAAAAAAAAAAAAAAAAAAAP////////////////////////8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/AwAAAAAAAAAAAAAAAAAAAAD/////////////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/wMAAAAAAAAAAAAAAAAAAAAA/////////////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP8DAAAAAAAAAAAAAAAAAAAAAP////////////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/AwAAAAAAAAAAAAAAAAAAAADw//////////////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/wMAAAAAAAAAAAAAAAAAAAAA8P//////////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP8DAAAAAAAAAAAAAAAAAAAAAPD//////////////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+P////////////////////8PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPj/////////////////////DwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4/////////////////////w8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+P////////////////////8DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPj/////////////////////AwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4/////////////////////wMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD/////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg/////////////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4P////////////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD/////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v//////////////////PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP7//////////////////z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+//////////////////8/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8P//////////////////DwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPD//////////////////w8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADw//////////////////8PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8P//////////////////DwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPD//////////////////w8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADw//////////////////8PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//////////////////DwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//////////////////w8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//////////////////8PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz/////////////////AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8/////////////////wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P////////////////8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAID///////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACA////////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP7//////////////38AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+//////////////9/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v//////////////fwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AMAgP///x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOADAID///8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAwCA////HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPD//x8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADw//8fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8P//HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMD//wcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADA//8HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwP//BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8/wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPz/AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8/wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/P8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPw/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/D8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPw/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/D8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPwPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8DwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/A8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPwDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        const [bytes, tileBytes] = await verifyBootstrap(compact);
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
        const southArea = cumulativeArea(south);
        const northArea = cumulativeArea(north);
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
        const pagePart = partCount <= 256 ? new Uint8Array(pageCount) : new Uint16Array(pageCount);
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
            pagePart.fill(number, nextPage, nextPage + count);
            nextPage += count;
        }
        requireBoundary(nextPage === pageCount, "대한민국 경계 조각 페이지 누락");
        return {bytes, data, bounds, gridWest, gridSouth, scale, columns, rows, tileIndex,
            tileCount, partCount, partOffset, pageCount, pagePart, southArea, northArea, polygons, rings, vertices,
            atlas: null, atlasRequested: false, atlasUnavailable: false, mixedChecks: 0};
    })().catch(error => {
        boundaryPromise = null;
        throw error;
    });
    return boundaryPromise;
}
function partNumber(boundary, page) {
    requireBoundary(Number.isInteger(page) && page >= 0 && page < boundary.pageCount,
        "대한민국 경계 페이지 범위 오류");
    return boundary.pagePart[page];
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
    if (boundary.atlas) {
        const refined = atlasState(boundary, point, tile);
        if (refined !== 2) return refined === 1;
    }
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
        const start = [page.data.getFloat64(40, true), page.data.getFloat64(48, true)];
        const end = [0, 0];
        for (let index = 1; index <= page.count; index++) {
            end[0] = page.data.getFloat64(40 + index * 16, true);
            end[1] = page.data.getFloat64(48 + index * 16, true);
            if (latitude >= Math.min(start[1], end[1]) && latitude <= Math.max(start[1], end[1])) {
                const side = orientation(start, end, location);
                if (side === 0 && longitude >= Math.min(start[0], end[0])
                    && longitude <= Math.max(start[0], end[0])) boundaryHit = true;
                if ((start[1] > latitude) !== (end[1] > latitude)) {
                    const crossing = end[1] > start[1] ? side > 0 : side < 0;
                    parity = parity !== crossing;
                }
            }
            start[0] = end[0];
            start[1] = end[1];
        }
    }
    requestAtlas(boundary);
    return boundaryHit || parity;
}
const remapWords = new DataView(new ArrayBuffer(8));
function remapTile(value, boundary) {
    remapWords.setBigUint64(0, value, false);
    const high = remapWords.getUint32(0, false);
    const low = remapWords.getUint32(4, false);
    const selector = high >>> 10;
    if (selector >= TILE_COUNT * 53) return null;
    const tile = boundary.tileIndex[Math.floor(selector / 53)];
    const vertical = Math.floor(tile / 512) * 8388608 + (high & 0x3ff) * 8192 + (low >>> 21) * 4 + 2;
    const horizontal = (tile % 512) * 8388608 + (low & 0x1fffff) * 4 + 2;
    return [vertical, horizontal];
}
export async function* sampleKorea(value) {
    requireBoundary(typeof value === "bigint" && value >= 0n && value <= MAX_U64,
        "64비트 난수 범위를 벗어났습니다.");
    const boundary = await loadBoundary();
    let candidateValue = value;
    for (let supplementalReads = 0; supplementalReads <= SUPPLEMENT_LIMIT; supplementalReads++) {
        const mapped = remapTile(candidateValue, boundary);
        if (mapped !== null) {
            const point = candidate(mapped, boundary);
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
                    const point = candidate(mapped, boundary);
                    const tile = tilePosition(boundary, point);
                    if (boundary.data.getUint32(128 + tile * 12, true) === 2
                        && (!boundary.atlas || atlasState(boundary, point, tile) === 2)) {
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
