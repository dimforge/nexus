import {generate} from './lib/user-guides.mjs';
const {pages, snippets} = generate();
console.log(`Generated ${pages} guide pages with ${snippets} extracted snippets.`);
