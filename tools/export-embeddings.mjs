// Calcule les embeddings de data/exemples.csv, avec le même modèle et le même préfixe que
// scripts/train.mjs, et les écrit dans .cache/tools/ :
// - embeddings.json : { texts, y, X } pour les scripts Python de tools/ ;
// - embeddings.csv : texte, camp, y (1 = droite), e0…e767, pour explorer les données ailleurs.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import * as transformers from '@huggingface/transformers';
import { createEmbedder } from '../site/classifier.js';

transformers.env.cacheDir = new URL('../.cache/models/', import.meta.url).pathname;
const out = new URL('../.cache/tools/', import.meta.url);
mkdirSync(out, { recursive: true });

const examples = readFileSync(new URL('../data/exemples.csv', import.meta.url), 'utf8')
    .trim()
    .split('\n')
    .slice(1)
    .filter((line) => line.trim() !== '')
    .map((line) => {
        const [camp, ...rest] = line.split(';');
        return { text: rest.join(';').trim(), y: camp.trim() === 'droite' ? 1 : 0 };
    });

const embed = await createEmbedder(transformers);
const X = [];
for (let i = 0; i < examples.length; i += 32) {
    X.push(...(await embed(examples.slice(i, i + 32).map((e) => e.text))));
}

writeFileSync(new URL('embeddings.json', out), JSON.stringify({ texts: examples.map((e) => e.text), y: examples.map((e) => e.y), X }));

const quote = (s) => `"${s.replaceAll('"', '""')}"`;
const header = ['texte', 'camp', 'y', ...X[0].map((_, i) => `e${i}`)].join(',');
const rows = examples.map((e, n) => [quote(e.text), e.y ? 'droite' : 'gauche', e.y, ...X[n].map((v) => +v.toPrecision(6))].join(','));
writeFileSync(new URL('embeddings.csv', out), [header, ...rows].join('\n') + '\n');

console.log(`${X.length} embeddings de dimension ${X[0].length} écrits dans .cache/tools/.`);
