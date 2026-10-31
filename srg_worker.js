import { convertNumber, formatOutput } from "./srg_random.js?v=1";
import { sampleKorea } from "./korea_boundary.js?v=6";

async function convertBatch({ runId, start, values }) {
    const blocks = new Array(values.length);
    let supplements = 0;
    const buffer = new BigUint64Array(32);
    let offset = buffer.length;
    for (let i = 0; i < values.length; i++) {
        const value = values[i];
        let recordSupplements = 0;
        const convert = convertNumber(value, sampleKorea);
        let step = await convert.next();
        while (!step.done) {
            if (recordSupplements >= 1024)
                throw new Error("대한민국 좌표 보완 난수 한도 초과");
            if (offset === buffer.length) {
                crypto.getRandomValues(buffer);
                offset = 0;
            }
            const word = buffer[offset++];
            supplements++;
            recordSupplements++;
            step = await convert.next({ value: word, bitsRemaining: 64 });
        }
        blocks[i] = formatOutput(step.value);
    }
    return { runId, start, blocks, supplements };
}
self.onmessage = async ({ data }) => {
    try {
        self.postMessage(await convertBatch(data));
    } catch (error) {
        self.postMessage({ runId: data?.runId, error: error instanceof Error ? error.message : String(error) });
    }
};
