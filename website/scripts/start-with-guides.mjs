import {watch} from 'node:fs';
import {spawn} from 'node:child_process';
import {generate, websiteRoot} from './lib/user-guides.mjs';
import path from 'node:path';

generate();
let timer;
const watchers = ['docs/user_guides/templates', 'docs-examples'].map(dir =>
  watch(path.join(websiteRoot, dir), {recursive: true}, (_, filename) => {
    if (!filename || !/\.(mdx|rs|py)$/.test(filename) || filename.includes('__pycache__')) return;
    clearTimeout(timer);
    timer = setTimeout(() => {
      try { console.log('User guides:', generate()); }
      catch (error) { console.error(error.message); }
    }, 150);
  }),
);
const child = spawn(process.execPath,
  [path.join(websiteRoot, 'node_modules/@docusaurus/core/bin/docusaurus.mjs'), 'start', ...process.argv.slice(2)],
  {cwd: websiteRoot, stdio: 'inherit'},
);
function close() { clearTimeout(timer); watchers.forEach(watcher => watcher.close()); }
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { close(); child.kill(signal); });
child.on('error', error => { close(); console.error(error); process.exitCode = 1; });
child.on('exit', code => { close(); process.exitCode = code ?? 0; });
