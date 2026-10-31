import { convertNumber, formatOutput } from "./srg_random.js?v=1";
import { sampleKorea } from "./korea_boundary.js?v=6";

const MASK64 = (1n << 64n) - 1n;
const SALT = 0xa5a5a5a5a5a5a5a5n;

function splitmix64(state) {
    state.value = (state.value + 0x9e3779b97f4a7c15n) & MASK64;
    let word = state.value;
    word = ((word ^ (word >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    word = ((word ^ (word >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return (word ^ (word >> 31n)) & MASK64;
}
async function convertBatch({ start, values, deterministic }) {
    const blocks = new Array(values.length);
    const reads = new Array(values.length);
    let supplements = 0;
    const buffer = new BigUint64Array(32);
    for (let i = 0; i < values.length; i++) {
        const value = values[i];
        const stream = { value: value ^ SALT };
        let offset = buffer.length;
        let recordSupplements = 0;
        const convert = convertNumber(value, sampleKorea);
        let step = await convert.next();
        while (!step.done) {
            if (recordSupplements >= 1024)
                throw new Error("대한민국 좌표 보완 난수 한도 초과");
            let word;
            if (deterministic) {
                word = splitmix64(stream);
            } else {
                if (offset === buffer.length) {
                    crypto.getRandomValues(buffer);
                    offset = 0;
                }
                word = buffer[offset++];
            }
            supplements++;
            recordSupplements++;
            step = await convert.next({ value: word, bitsRemaining: 64 });
        }
        blocks[i] = formatOutput(step.value);
        reads[i] = recordSupplements;
    }
    return { start, blocks, supplements, reads };
}
self.onmessage = async ({ data }) => {
    try {
        self.postMessage(await convertBatch(data));
    } catch (error) {
        self.postMessage({ error: error instanceof Error ? error.message : String(error) });
    }
};
