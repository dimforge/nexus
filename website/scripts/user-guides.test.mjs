import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {specialize, extractSnippets, generate} from './lib/user-guides.mjs';

test('inline and multiline conditions preserve shared MDX, including nested tags', () => {
  const source = 'Use <rust>`State::new`</rust><python>`State()`</python>.\n<rust>Rust <rust>nested</rust></rust>\n<Tabs />';
  assert.equal(specialize(source, 'rust'), 'Use `State::new`.\nRust nested\n<Tabs />');
  assert.equal(specialize(source, 'python'), 'Use `State()`.\n\n<Tabs />');
  assert.throws(() => specialize('<rust>broken</python>', 'rust'), /Mismatched/);
  assert.throws(() => specialize('<python>broken', 'rust'), /Unclosed/);
});

test('Rust and Python regions deindent, handle CRLF, and strip nested markers', () => {
  const source = '    // DOCUSAURUS: outer start\r\n    first();\r\n    // DOCUSAURUS: inner start\r\n        second();\r\n    // DOCUSAURUS: inner stop\r\n    // DOCUSAURUS: outer stop';
  const snippets = extractSnippets(source);
  assert.equal(snippets.get('outer'), 'first();\n    second();');
  assert.equal(snippets.get('inner'), 'second();');
  assert.equal(extractSnippets('# DOCUSAURUS: py start\n    call()\n# DOCUSAURUS: py stop').get('py'), 'call()');
  assert.throws(() => extractSnippets('// DOCUSAURUS: a start'), /unclosed/);
  assert.throws(() => extractSnippets('// DOCUSAURUS: a stop'), /mismatched/);
  assert.throws(() => extractSnippets('// DOCUSAURUS: a start\nx\n// DOCUSAURUS: a stop\n// DOCUSAURUS: a start'), /duplicate/);
});

test('generation fails before modifying outputs, and removes obsolete pages on success', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'nexus-guide-'));
  const put = (file, text) => { const dest = path.join(root, file); mkdirSync(path.dirname(dest), {recursive: true}); writeFileSync(dest, text); };
  try {
    put('docs-examples/example.rs', '// DOCUSAURUS: example start\nlet x = 1;\n// DOCUSAURUS: example stop');
    put('docs/user_guides/templates/page.mdx', "<rust><load path='example.rs' marker='missing' /></rust>");
    put('docs/user_guides/rust/old.mdx', 'old');
    assert.throws(() => generate(root), /missing snippet/);
    assert.equal(readFileSync(path.join(root, 'docs/user_guides/rust/old.mdx'), 'utf8'), 'old');
    put('docs/user_guides/templates/page.mdx', "Shared <rust><load path='example.rs' marker='example' /></rust><python>Python.</python>");
    assert.deepEqual(generate(root), {pages: 3, snippets: 2});
    assert.equal(readFileSync(path.join(root, 'docs/user_guides/all/page.mdx'), 'utf8'), 'Shared <span className="guide-language-rust" title="Rust">let x = 1;</span><span className="guide-language-python" title="Python">Python.</span>');
    put('docs/user_guides/templates/bad.mdx', '```rust\nlet unchecked = 1;\n```');
    assert.throws(() => generate(root), /must load a checked source/);
    rmSync(path.join(root, 'docs/user_guides/templates/bad.mdx'));
    assert.equal(existsSync(path.join(root, 'docs/user_guides/rust/old.mdx')), false);
    assert.equal(readFileSync(path.join(root, 'docs/user_guides/rust/page.mdx'), 'utf8'), 'Shared let x = 1;');
    assert.equal(readFileSync(path.join(root, 'docs/user_guides/python/page.mdx'), 'utf8'), 'Shared Python.');
  } finally { rmSync(root, {recursive: true, force: true}); }
});

test('All mode labels both languages and resolves nested conditions in their outer language', () => {
  const source = '<rust>Rust</rust><python><rust>hidden</rust><python>Python</python></python>';
  assert.equal(specialize(source, 'all'), '<span className="guide-language-rust" title="Rust">Rust</span><span className="guide-language-python" title="Python">Python</span>');
  assert.equal(specialize('<rust>\n\n## Rust section\n\n</rust>', 'all'), '\n\n<div className="guide-language-rust" title="Rust">\n\n\n\n## Rust section\n\n\n\n</div>\n\n');
  assert.throws(() => specialize('<rust>broken</python>', 'all'), /Mismatched/);
});
