export async function convertParallel(values, { workers = 4, deterministic = false, onProgress, signal } = {}) {
    if (!Array.isArray(values) || values.some(value => typeof value !== "bigint"))
        throw new TypeError("64비트 난수 배열이 필요합니다.");
    if (!values.length) return { blocks: [], supplements: 0, reads: [] };
    if (signal?.aborted) throw new DOMException("작업이 중단됐습니다.", "AbortError");
    const batchSize = 128;
    const count = Math.min(Math.max(1, Math.trunc(workers) || 1), Math.ceil(values.length / batchSize), 4);
    const pool = [];
    const blocks = new Array(values.length);
    const reads = new Array(values.length);
    let completed = 0;
    let supplementalReads = 0;
    let nextStart = 0;
    let finishedWorkers = 0;
    let settled = false;
    let rejectPending;
    const abort = () => rejectPending?.(new DOMException("작업이 중단됐습니다.", "AbortError"));
    try {
        const result = await new Promise((resolve, reject) => {
            rejectPending = error => {
                if (settled) return;
                settled = true;
                reject(error);
            };
            const dispatch = worker => {
                if (nextStart >= values.length) {
                    finishedWorkers++;
                    if (finishedWorkers === pool.length && !settled) {
                        settled = true;
                        resolve({ blocks, supplements: supplementalReads, reads });
                    }
                    return;
                }
                const start = nextStart;
                const end = Math.min(values.length, start + batchSize);
                nextStart = end;
                worker.postMessage({ start, values: values.slice(start, end), deterministic });
            };
            try {
                for (let i = 0; i < count; i++)
                    pool.push(new Worker(new URL("./srg_worker.js?v=1", import.meta.url), { type: "module" }));
                for (const worker of pool) {
                    worker.onerror = event => rejectPending(new Error(event.message || "Web Worker 실행 오류"));
                    worker.onmessageerror = () => rejectPending(new Error("Web Worker 결과 해석 오류"));
                    worker.onmessage = ({ data }) => {
                        if (settled) return;
                        if (typeof data?.error === "string") {
                            rejectPending(new Error(data.error));
                            return;
                        }
                        if (!Array.isArray(data?.blocks) || data.blocks.length === 0) {
                            rejectPending(new Error("Web Worker 응답 형식 오류"));
                            return;
                        }
                        for (let i = 0; i < data.blocks.length; i++) {
                            blocks[data.start + i] = data.blocks[i];
                            reads[data.start + i] = data.reads[i];
                        }
                        completed += data.blocks.length;
                        supplementalReads += data.supplements;
                        if (onProgress) {
                            try { onProgress(completed, supplementalReads); }
                            catch (error) { rejectPending(error); return; }
                        }
                        dispatch(worker);
                    };
                }
                if (signal) signal.addEventListener("abort", abort, { once: true });
                for (const worker of pool) dispatch(worker);
            } catch (error) {
                rejectPending(error);
            }
        });
        return result;
    } finally {
        if (signal) signal.removeEventListener("abort", abort);
        for (const worker of pool) worker.terminate();
    }
}
