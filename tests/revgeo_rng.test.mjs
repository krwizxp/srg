import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { webcrypto } from 'node:crypto';
import { test } from 'node:test';
import vm from 'node:vm';

const html = readFileSync(new URL('../revgeo.html', import.meta.url), 'utf8');
const generator = html.slice(html.indexOf('async function generateRandomValues('), html.indexOf('function loadManualRng()'));
const signal = () => new AbortController().signal;
function setup(overrides = {}) {
  const context = vm.createContext({
    crypto: webcrypto,
    fmt: String,
    // Every entropy request must stay on the device, including failure paths.
    fetch() { assert.fail('must not request server entropy'); },
    Math: Object.assign(Object.create(Math), { random() { assert.fail('must not use Math.random'); } }),
    ...overrides,
  });
  vm.runInContext(`const GENERATION_LIMIT = 10000;\n${generator}`, context);
  return context.generateRandomValues;
}

test('splits the 10,000-value Web Crypto request at its 65,536-byte quota', async () => {
  const calls = [];
  const generate = setup({ crypto: { getRandomValues(words) {
    calls.push(words.byteLength);
    assert.ok(words.byteLength <= 65536);
    words.fill(0xffffffff);
    return words;
  } } });
  const values = await generate(10000, signal());
  assert.deepEqual(calls, [65536, 14464]);
  assert.equal(values.length, 10000);
  assert.equal(values[0], (1n << 64n) - 1n);
  assert.equal(values.at(-1), (1n << 64n) - 1n);
});

test('preserves zero, values above MAX_SAFE_INTEGER and u64 max from client entropy', async () => {
  const generate = setup({ crypto: { getRandomValues(words) {
    words.set([0, 0, 0x200000, 1, 0xffffffff, 0xffffffff]);
    return words;
  } } });
  const values = await generate(3, signal());
  assert.deepEqual(Array.from(values), [0n, 9007199254740993n, 18446744073709551615n]);
});

test('uses actual Web Crypto and keeps all u64 values in range', async () => {
  const values = await setup()(64, signal());
  assert.equal(values.length, 64);
  assert.ok(values.every(value => value >= 0n && value <= (1n << 64n) - 1n));
  assert.ok(new Set(values).size > 1);
});

test('uses client entropy regardless of page host or obsolete backend metadata', async () => {
  for (const href of ['https://example.com/revgeo.html', 'http://127.0.0.1:8787/revgeo.html']) {
    let calls = 0;
    const generate = setup({
      location: { href },
      query: () => ({ content: '/api/random' }),
      crypto: { getRandomValues(words) { calls++; words.fill(7); return words; } },
    });
    const values = await generate(2, signal());
    assert.equal(calls, 1);
    assert.deepEqual(Array.from(values), [30064771079n, 30064771079n]);
  }
});

test('rejects invalid counts before requesting any entropy', async () => {
  const generate = setup({ crypto: { getRandomValues() { assert.fail('must not request entropy'); } } });
  for (const count of [0, -1, 1.5, 10001, NaN, Infinity])
    await assert.rejects(generate(count, signal()));
});

test('fails without a fallback when client cryptography is unavailable', async () => {
  for (const crypto of [undefined, null, {}])
    await assert.rejects(setup({ crypto })(1, signal()), /이 기기의 브라우저에서 암호화 난수/);
});

test('propagates client entropy errors without a server or insecure fallback', async () => {
  const generate = setup({ crypto: { getRandomValues() { throw new Error('client entropy failure'); } } });
  await assert.rejects(generate(1, signal()), /client entropy failure/);
});

test('does not request entropy after cancellation', async () => {
  const controller = new AbortController();
  controller.abort();
  const generate = setup({ crypto: { getRandomValues() { assert.fail('must not request entropy'); } } });
  await assert.rejects(generate(1, controller.signal), { name: 'AbortError' });
});
