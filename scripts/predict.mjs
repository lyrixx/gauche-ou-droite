// Teste le classifieur en ligne de commande : node scripts/predict.mjs "facho" "le vélo" ...
import { readFileSync } from 'node:fs';
import * as transformers from '@huggingface/transformers';
import { classify, createEmbedder } from '../site/classifier.js';

transformers.env.cacheDir = new URL('../.cache/models/', import.meta.url).pathname;

const classifier = JSON.parse(readFileSync(new URL('../site/classifier.json', import.meta.url), 'utf8'));
const embed = await createEmbedder(transformers);

for (const text of process.argv.slice(2)) {
    const { pDroite, source } = await classify(classifier, embed, text);
    const camp = pDroite >= 0.5 ? 'droite' : 'gauche';
    const confidence = Math.round((pDroite >= 0.5 ? pDroite : 1 - pDroite) * 100);
    console.log(`${camp} ${String(confidence).padStart(3)} %  (${source.padEnd(7)})  ${text}`);
}
