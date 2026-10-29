import {boundedBytes, loadCoordinateCore} from "./coordinates_wasm.js";

const MAX_U64 = (1n << 64n) - 1n;
const INDEX_SHA256 = "8cb20a5c0da9c3d6225efb0323e24a4f992eb3029a772da3a71a55d3f503bf17";
const TILE_SHA256 = "492fdbeee7e7a64816d7ea90c57cb5773685fbf96e0b95911a8b7cdcfbf2ae8b";
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
        const [bytes, tileBytes, core] = await Promise.all([
            download("index.bin", MAX_INDEX_BYTES, INDEX_SHA256),
            download("tile-select-v5.bin", 32768, TILE_SHA256),
            loadCoordinateCore(),
        ]);
        core.setTiles(tileBytes);
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
        return {bytes, data, bounds, gridWest, gridSouth, scale, columns, rows, core,
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
        const raw = boundary.core.inflate(part.bytes.subarray(start, start + length), rawLength);
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
    let parity = false;
    let boundaryHit = false;
    for await (const page of tilePages(boundary, tile)) {
        const relation = boundary.core.pageRelation(page, latitude, longitude);
        parity = parity !== Boolean(relation & 1);
        boundaryHit ||= Boolean(relation & 2);
    }
    return boundaryHit || parity;
}
export async function* sampleKorea(value) {
    requireBoundary(typeof value === "bigint" && value >= 0n && value <= MAX_U64,
        "64비트 난수 범위를 벗어났습니다.");
    const boundary = await loadBoundary();
    let candidateValue = value;
    for (let supplementalReads = 0; supplementalReads <= SUPPLEMENT_LIMIT; supplementalReads++) {
        const point = boundary.core.candidate(candidateValue, boundary.bounds);
        if (point !== null && await covers(boundary, point)) return point;
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
                const point = boundary.core.candidate(current, boundary.bounds);
                if (point !== null) {
                    const tile = tilePosition(boundary, point);
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

export async function loadCoordinates() {
    const core = await loadCoordinateCore();
    return {sampleKorea, prepareKorea, worldCoordinates: core.world};
}
