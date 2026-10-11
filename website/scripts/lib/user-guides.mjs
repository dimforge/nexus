import {readFileSync, readdirSync, mkdirSync, writeFileSync, rmSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export const websiteRoot = fileURLToPath(new URL('../../', import.meta.url));
export const languages = ['rust', 'python'];
export const variants = [...languages, 'all'];

// These tags may occur inside a sentence, or surround a whole MDX section.
export function specialize(source, language) {
  if (!variants.includes(language)) throw new Error(`Unknown language: ${language}`);
  const stack = [];
  let result = '';
  let cursor = 0;
  let wrapper;
  const tokens = [...source.matchAll(/<(\/?)(rust|python)>/g)];
  for (const [index, match] of tokens.entries()) {
    if (stack.every(tag => tag === (language === 'all' ? stack[0] : language))) {
      result += source.slice(cursor, match.index);
    }
    const [, closing, tag] = match;
    if (closing) {
      if (stack.pop() !== tag) throw new Error(`Mismatched closing tag: ${match[0]}`);
      if (language === 'all' && stack.length === 0) result += wrapper === 'div' ? '\n\n</div>\n\n' : '</span>';
    } else {
      if (language === 'all' && stack.length === 0) {
        let depth = 1;
        const end = tokens.slice(index + 1).find(token => {
          depth += token[1] ? -1 : 1;
          return depth === 0;
        });
        const content = source.slice(match.index + match[0].length, end?.index);
        wrapper = /\n\s*\n/.test(content) ? 'div' : 'span';
        if (wrapper === 'div') result += '\n\n';
        result += `<${wrapper} className="guide-language-${tag}" title="${tag === 'rust' ? 'Rust' : 'Python'}">`;
        if (wrapper === 'div') result += '\n\n';
      }
      stack.push(tag);
    }
    cursor = match.index + match[0].length;
  }
  if (stack.length) throw new Error(`Unclosed language tag: <${stack.at(-1)}>`);
  result += source.slice(cursor);
  if (/<\/?(?:rust|python)\b/.test(result)) throw new Error('Malformed language tag');
  return result;
}

export function extractSnippets(source, filename = '<source>') {
  const snippets = new Map();
  const stack = [];
  const lines = source.replace(/\r\n/g, '\n').split('\n');
  for (const [i, line] of lines.entries()) {
    const marker = line.match(/^\s*(?:\/\/|#) DOCUSAURUS: ([\w-]+) (start|stop)\s*$/);
    if (!marker) {
      if (/^\s*(?:\/\/|#) DOCUSAURUS:/.test(line)) throw new Error(`${filename}:${i + 1}: malformed marker`);
      for (const frame of stack) frame.lines.push(line);
      continue;
    }
    const [, name, action] = marker;
    if (action === 'start') {
      if (snippets.has(name) || stack.some(frame => frame.name === name)) {
        throw new Error(`${filename}:${i + 1}: duplicate marker ${name}`);
      }
      stack.push({name, lines: []});
    } else {
      const frame = stack.pop();
      if (!frame || frame.name !== name) throw new Error(`${filename}:${i + 1}: mismatched stop ${name}`);
      const nonempty = frame.lines.filter(line => line.trim());
      if (!nonempty.length) throw new Error(`${filename}: empty snippet ${name}`);
      const indent = Math.min(...nonempty.map(line => line.match(/^\s*/)[0].length));
      snippets.set(name, frame.lines.map(line => line.slice(indent)).join('\n').trimEnd());
    }
  }
  if (stack.length) throw new Error(`${filename}: unclosed marker ${stack.at(-1).name}`);
  return snippets;
}

function sourceFiles(dir) {
  return readdirSync(dir, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name)).flatMap(entry => {
    if (entry.name.startsWith('.') || ['target', '__pycache__'].includes(entry.name)) return [];
    const file = path.join(dir, entry.name);
    return entry.isDirectory() ? sourceFiles(file) : [file];
  });
}

export function generate(root = websiteRoot) {
  const examples = path.join(root, 'docs-examples');
  const templates = path.join(root, 'docs/user_guides/templates');
  const sources = new Map();
  for (const file of sourceFiles(examples).filter(file => /\.(rs|py)$/.test(file))) {
    sources.set(path.relative(examples, file).split(path.sep).join('/'), extractSnippets(readFileSync(file, 'utf8'), file));
  }
  const outputs = new Map();
  let loads = 0;
  for (const language of variants) {
    for (const file of sourceFiles(templates).filter(file => file.endsWith('.mdx'))) {
      let page = specialize(readFileSync(file, 'utf8'), language);
      for (const block of page.matchAll(/```(?:rust|python)\s*\n([\s\S]*?)```/g)) {
        if (!/^<load path='[^']+' marker='[\w-]+'\s*\/>\s*$/.test(block[1])) {
          throw new Error(`${file}: Rust/Python code blocks must load a checked source region`);
        }
      }
      page = page.replace(/<load path='([^']+)' marker='([\w-]+)'\s*\/>/g, (_, sourcePath, marker) => {
        const snippet = sources.get(sourcePath)?.get(marker);
        if (!snippet) throw new Error(`${file} (${language}): missing snippet ${sourcePath}#${marker}`);
        loads++;
        return snippet;
      });
      if (/<load\b/.test(page)) throw new Error(`${file}: malformed <load> tag`);
      // Every Rust/Python code block must come from checked sources.
      if (/DOCUSAURUS:/.test(page)) throw new Error(`${file}: leaked source marker`);
      outputs.set(path.join(root, 'docs/user_guides', language, path.basename(file)), page);
    }
  }
  // Validate all inputs before replacing any generated output. Only these three
  // dedicated generated directories are removed; templates are never touched.
  for (const language of variants) {
    const dir = path.join(root, 'docs/user_guides', language);
    rmSync(dir, {recursive: true, force: true});
    mkdirSync(dir, {recursive: true});
  }
  for (const [file, page] of outputs) writeFileSync(file, page);
  return {pages: outputs.size, snippets: loads};
}
