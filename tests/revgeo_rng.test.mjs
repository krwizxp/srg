import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { webcrypto } from 'node:crypto';
import { test } from 'node:test';
import vm from 'node:vm';

const html = readFileSync(new URL('../revgeo.html', import.meta.url), 'utf8');
const generator = html.slice(html.indexOf('const RANDOM_SOURCE_LABELS'), html.indexOf('function loadManualRng()'));
const parser = html.slice(html.indexOf('function parseU64('), html.indexOf('const bitsView ='));
const signal = () => new AbortController().signal;
function setup(overrides = {}) {
  const context = vm.createContext({
    crypto: webcrypto, AbortSignal, URL,
    location: { href: 'http://127.0.0.1:8787/revgeo.html' },
    query: () => ({ content: '/api/random' }),
    fmt: String, ...overrides,
  });
  vm.runInContext(`const GENERATION_LIMIT = 10000;
    const U64_MAX = (1n << 64n) - 1n;
    const U64_TEXT_LIMIT = 64;
    const U64_BOUND_TEXT = 'u64';
    const U64_FORMAT_ERROR = 'invalid u64';
    ${parser}\n${generator}`, context);
  return context.generateRandomValues;
}

test('Web Crypto splits the 10,000-value request at its 65,536-byte quota', async () => {
  const calls = [];
  const generate = setup({ crypto: { getRandomValues(words) {
    calls.push(words.byteLength);
    assert.ok(words.byteLength <= 65536);
    words.fill(0xffffffff);
    return words;
  } } });
  const result = await generate(10000, 'browser', signal());
  assert.deepEqual(calls, [65536, 14464]);
  assert.equal(result.values.length, 10000);
  assert.equal(result.values[0], (1n << 64n) - 1n);
  assert.equal(result.values.at(-1), (1n << 64n) - 1n);
});

test('uses actual Web Crypto and keeps u64 values in range', async () => {
  const result = await setup()(64, 'browser', signal());
  assert.equal(result.source, 'web-crypto');
  assert.ok(result.values.every(value => value >= 0n && value <= (1n << 64n) - 1n));
  assert.ok(new Set(result.values).size > 1);
});

test('preserves u64 max, zero and values above Number.MAX_SAFE_INTEGER from Rust', async () => {
  const generate = setup({ fetch: async (url, options) => {
    assert.equal(String(url), 'http://127.0.0.1:8787/api/random');
    assert.equal(options.method, 'POST');
    assert.equal(options.body, '3');
    assert.equal(options.redirect, 'error');
    return { ok: true, json: async () => ({ source: 'rdseed', values: ['0', '9007199254740993', '18446744073709551615'] }) };
  } });
  const result = await generate(3, 'srg', signal());
  assert.equal(result.values[0], 0n);
  assert.equal(result.values[1], 9007199254740993n);
  assert.equal(result.values[2], 18446744073709551615n);
  assert.equal(result.source, 'rdseed');
});

test('rejects invalid counts before requesting any entropy', async () => {
  const generate = setup({ crypto: { getRandomValues() { assert.fail('must not request entropy'); } } });
  for (const count of [0, -1, 1.5, 10001, NaN, Infinity])
    await assert.rejects(generate(count, 'browser', signal()));
});

test('fails explicitly when cryptography or the local backend is unavailable', async () => {
  await assert.rejects(setup({ crypto: undefined })(1, 'browser', signal()), /암호화 난수/);
  await assert.rejects(setup({ query: () => null })(1, 'srg', signal()), /serve-revgeo/);
  await assert.rejects(setup({ fetch: async () => ({ ok: false, status: 503, text: async () => 'RDRAND 실패' }) })(1, 'srg', signal()), /503.*RDRAND/);
});

test('rejects malformed, rounded, out-of-range and mislabeled backend responses', async () => {
  for (const data of [
    null,
    { source: 'toString', values: ['1'] },
    { source: 'web-crypto', values: ['1'] },
    { source: 'rdseed', values: [] },
    { source: 'rdseed', values: [9007199254740993] },
    { source: 'rdseed', values: ['18446744073709551616'] },
    { source: 'rdseed', values: ['-1'] },
    { source: 'rdseed', values: ['1.2'] },
  ]) {
    await assert.rejects(setup({ fetch: async () => ({ ok: true, json: async () => data }) })(1, 'srg', signal()));
  }
});

test('does not generate after cancellation', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(setup()(1, 'browser', controller.signal), { name: 'AbortError' });
});
