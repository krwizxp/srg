const MAX_BYTES = 1_000_000;
const MAX_PAGE = 15_876;
const MAX_U64 = (1n << 64n) - 1n;
let boundaryPromise;
const pages = new Map();
const parts = new Map();
async function boundedBytes(response, maximum) {
    const reader = response.body.getReader();
    const chunks = [];
    let size = 0;
    try {
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            size += value.length;
            if (size > maximum) throw new Error("대한민국 경계 데이터 크기 한도를 초과했습니다.");
            chunks.push(value);
        }
    } catch (error) {
        await reader.cancel().catch(() => {});
        throw error;
    } finally {
        reader.releaseLock();
    }
    const result = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
    }
    return result;
}
async function loadBoundary() {
    if (!boundaryPromise) {
        boundaryPromise = (async () => {
            const response = await fetch(new URL("./data/korea-boundary-v1/index.bin", import.meta.url));
            if (!response.ok) throw new Error(`대한민국 경계 데이터 불러오기 실패: HTTP ${response.status}`);
            const bytes = await boundedBytes(response, MAX_BYTES);
            const data = new DataView(bytes.buffer);
            if (bytes.length < 32 || new TextDecoder().decode(bytes.subarray(0, 8)) !== "SRGKR001")
                throw new Error("대한민국 경계 데이터 버전 오류");
            const blocks = data.getUint32(8, true);
            const triangles = data.getUint32(12, true);
            if (!blocks || blocks !== Math.ceil(triangles / 256) || 32 + blocks * 24 !== bytes.length || data.getUint32(16, true) !== 14 || !data.getUint32(20, true))
                throw new Error("대한민국 경계 데이터 인덱스 오류");
            let previous = -1n;
            let previousOffset = 0;
            let previousPart = 0;
            for (let i = 0; i < blocks; i++) {
                const entry = 32 + i * 24;
                const end = data.getBigUint64(entry, true);
                const offset = data.getUint32(entry + 8, true);
                const part = data.getUint32(entry + 12, true);
                const count = data.getUint32(entry + 16, true);
                const length = data.getUint32(entry + 20, true);
                if (end <= previous || !(part === previousPart && offset === previousOffset || part === previousPart + 1 && offset === 0) || !count || count > 256 || !length || offset + length > 524288)
                    throw new Error("대한민국 경계 데이터 블록 인덱스 오류");
                previous = end;
                previousOffset = offset + length;
                previousPart = part;
            }
            if (previous !== MAX_U64 || previousPart + 1 !== data.getUint32(20, true))
                throw new Error("대한민국 경계 데이터 범위 오류");
            return { bytes, data, blocks };
        })().catch(error => {
            boundaryPromise = null;
            throw error;
        });
    }
    return boundaryPromise;
}
async function loadPart(part) {
    if (parts.has(part)) {
        const pending = parts.get(part);
        parts.delete(part);
        parts.set(part, pending);
        return pending;
    }
    const pending = (async () => {
        const name = String(part).padStart(3, "0");
        const response = await fetch(new URL(`./data/korea-boundary-v1/part-${name}.bin`, import.meta.url));
        if (!response.ok) throw new Error(`대한민국 경계 블록 불러오기 실패: HTTP ${response.status}`);
        return boundedBytes(response, 524288);
    })();
    parts.set(part, pending);
    if (parts.size > 8) parts.delete(parts.keys().next().value);
    try { return await pending; }
    catch (error) {
        if (parts.get(part) === pending) parts.delete(part);
        throw error;
    }
}
async function loadPage(boundary, block) {
    if (pages.has(block)) {
        const page = pages.get(block);
        pages.delete(block);
        pages.set(block, page);
        return page;
    }
    const pending = (async () => {
        const entry = 32 + block * 24;
        const offset = boundary.data.getUint32(entry + 8, true);
        const part = boundary.data.getUint32(entry + 12, true);
        const partBytes = await loadPart(part);
        const length = boundary.data.getUint32(entry + 20, true);
        const stream = new Blob([partBytes.subarray(offset, offset + length)]).stream().pipeThrough(new DecompressionStream("deflate"));
        const shuffled = await boundedBytes(new Response(stream), MAX_PAGE);
        const bytes = new Uint8Array(shuffled.length);
        bytes.set(shuffled.subarray(0, 4));
        const data = new DataView(bytes.buffer);
        if (bytes.length < 4) throw new Error("대한민국 경계 데이터 블록 길이 오류");
        const vertices = data.getUint16(0, true);
        const count = data.getUint16(2, true);
        const records = 4 + vertices * 16;
        if (!vertices || vertices > 768 || count !== boundary.data.getUint32(entry + 16, true) || bytes.length !== records + count * 14)
            throw new Error("대한민국 경계 데이터 블록 오류");
        for (let word = 0; word < vertices * 2; word++)
            for (let lane = 0; lane < 8; lane++)
                bytes[4 + word * 8 + lane] = shuffled[4 + lane * vertices * 2 + word];
        for (let row = 0; row < count; row++)
            for (let lane = 0; lane < 14; lane++)
                bytes[records + row * 14 + lane] = shuffled[records + lane * count + row];
        let previous = block ? boundary.data.getBigUint64(entry - 24, true) : -1n;
        for (let i = 0; i < count; i++) {
            const record = records + i * 14;
            const end = previous + data.getBigUint64(record, true);
            if (end <= previous) throw new Error("대한민국 경계 데이터 가중치 오류");
            data.setBigUint64(record, end, true);
            for (let corner = 0; corner < 3; corner++)
                if (data.getUint16(record + 8 + corner * 2, true) >= vertices)
                    throw new Error("대한민국 경계 데이터 정점 오류");
            previous = end;
        }
        if (previous !== boundary.data.getBigUint64(entry, true)) throw new Error("대한민국 경계 데이터 블록 범위 오류");
        for (let i = 0; i < vertices; i++) {
            const latitude = data.getFloat64(4 + i * 16, true);
            const longitude = data.getFloat64(12 + i * 16, true);
            if (!(latitude >= 33 && latitude < 39 && longitude >= 124 && longitude < 132))
                throw new Error("대한민국 경계 데이터 좌표 오류");
        }
        return { data, records, count };
    })();
    pages.set(block, pending);
    if (pages.size > 128) pages.delete(pages.keys().next().value);
    try {
        return await pending;
    } catch (error) {
        if (pages.get(block) === pending) pages.delete(block);
        throw error;
    }
}
function reverseBits(value) {
    let result = value;
    for (const [shift, mask] of [[1n, 0x5555555555555555n], [2n, 0x3333333333333333n], [4n, 0x0f0f0f0f0f0f0f0fn], [8n, 0x00ff00ff00ff00ffn], [16n, 0x0000ffff0000ffffn], [32n, 0x00000000ffffffffn]])
        result = ((result >> shift) & mask) | ((result & mask) << shift);
    return result;
}
export async function sampleKorea(value) {
    if (typeof value !== "bigint" || value < 0n || value > MAX_U64) throw new RangeError("64비트 난수 범위를 벗어났습니다.");
    const boundary = await loadBoundary();
    let low = 0;
    let high = boundary.blocks;
    while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (boundary.data.getBigUint64(32 + middle * 24, true) < value) low = middle + 1;
        else high = middle;
    }
    const block = low;
    const page = await loadPage(boundary, block);
    low = 0;
    high = page.count;
    while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (page.data.getBigUint64(page.records + middle * 14, true) < value) low = middle + 1;
        else high = middle;
    }
    const record = page.records + low * 14;
    const end = page.data.getBigUint64(record, true);
    const begin = low ? page.data.getBigUint64(record - 14, true) + 1n : block ? boundary.data.getBigUint64(32 + (block - 1) * 24, true) + 1n : 0n;
    const offset = value - begin;
    const width = end - begin + 1n;
    let first = Math.max(Number.EPSILON / 2, Math.min(1 - Number.EPSILON / 2, (Number(offset) + 0.5) / Number(width)));
    let second = Number(reverseBits(offset) >> 12n) / 4_503_599_627_370_496 + 1 / 9_007_199_254_740_992;
    if (first + second > 1) {
        first = 1 - first;
        second = 1 - second;
    }
    const weights = [1 - first - second, first, second].map(weight => weight * (1 - 3 / 1_048_576) + 1 / 1_048_576);
    let latitude = 0;
    let longitude = 0;
    for (let corner = 0; corner < 3; corner++) {
        const vertex = page.data.getUint16(record + 8 + corner * 2, true);
        latitude += page.data.getFloat64(4 + vertex * 16, true) * weights[corner];
        longitude += page.data.getFloat64(12 + vertex * 16, true) * weights[corner];
    }
    return [latitude, longitude];
}
