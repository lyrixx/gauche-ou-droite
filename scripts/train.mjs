// Entraîne le classifieur gauche/droite et écrit site/classifier.json.
//
// 1. Transforme chaque exemple de data/exemples.csv en vecteur (embedding),
//    avec exactement le même modèle que celui utilisé dans le navigateur.
// 2. Entraîne une régression logistique sur ces vecteurs.
// 3. Mesure la précision en validation croisée (sur des exemples jamais vus).
// 4. Sauvegarde les poids, et la liste des exemples pour y répondre directement,
//    dans un JSON que le site charge.

import { readFileSync, writeFileSync } from 'node:fs';
import { env, pipeline } from '@huggingface/transformers';
import { EMBEDDING_MODEL, PREFIX, DTYPE } from '../site/config.js';
import { normalize } from '../site/classifier.js';

// Garde le modèle téléchargé en local (et dans le cache de la CI) pour ne pas le retélécharger.
env.cacheDir = new URL('../.cache/models/', import.meta.url).pathname;

const LAMBDA = 0.01; // régularisation L2 : évite de "coller" aux exemples
const EPOCHS = 2000;
const LEARNING_RATE = 0.5;
const FOLDS = 5;

const examples = readFileSync(new URL('../data/exemples.csv', import.meta.url), 'utf8')
    .trim()
    .split('\n')
    .slice(1)
    .filter((line) => line.trim() !== '')
    .map((line) => {
        const [camp, ...rest] = line.split(';');
        return { text: rest.join(';').trim(), y: camp.trim() === 'droite' ? 1 : 0 };
    });

console.log(`${examples.length} exemples (${examples.filter((e) => e.y === 0).length} gauche, ${examples.filter((e) => e.y === 1).length} droite)`);

console.log(`Chargement de ${EMBEDDING_MODEL}...`);
const extractor = await pipeline('feature-extraction', EMBEDDING_MODEL, { dtype: DTYPE });

console.log('Calcul des embeddings...');
const output = await extractor(examples.map((e) => PREFIX + e.text), { pooling: 'mean', normalize: true });
const X = output.tolist();
const Y = examples.map((e) => e.y);
const dim = X[0].length;

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

function standardize(rows) {
    const mean = new Array(dim).fill(0);
    const std = new Array(dim).fill(0);
    for (const r of rows) r.forEach((v, i) => (mean[i] += v / rows.length));
    for (const r of rows) r.forEach((v, i) => (std[i] += (v - mean[i]) ** 2 / rows.length));
    return { mean, std: std.map((v) => Math.sqrt(v) || 1) };
}

function train(rows, labels) {
    const { mean, std } = standardize(rows);
    const Z = rows.map((r) => r.map((v, i) => (v - mean[i]) / std[i]));
    const w = new Array(dim).fill(0);
    let b = 0;
    for (let epoch = 0; epoch < EPOCHS; epoch++) {
        const gw = w.map((wi) => LAMBDA * wi);
        let gb = 0;
        Z.forEach((z, n) => {
            const err = sigmoid(dot(w, z) + b) - labels[n];
            for (let i = 0; i < dim; i++) gw[i] += (err * z[i]) / Z.length;
            gb += err / Z.length;
        });
        for (let i = 0; i < dim; i++) w[i] -= LEARNING_RATE * gw[i];
        b -= LEARNING_RATE * gb;
    }
    return { mean, std, weights: w, bias: b };
}

function predict(model, row) {
    const z = row.map((v, i) => (v - model.mean[i]) / model.std[i]);
    return sigmoid(dot(model.weights, z) + model.bias);
}

// Validation croisée : on entraîne sur 4/5 des exemples, on teste sur le 1/5 restant.
const order = examples.map((_, i) => i).sort(() => Math.random() - 0.5);
let correct = 0;
const errors = [];
for (let f = 0; f < FOLDS; f++) {
    const test = order.filter((_, k) => k % FOLDS === f);
    const trainIdx = order.filter((_, k) => k % FOLDS !== f);
    const model = train(trainIdx.map((i) => X[i]), trainIdx.map((i) => Y[i]));
    for (const i of test) {
        const p = predict(model, X[i]);
        if ((p >= 0.5 ? 1 : 0) === Y[i]) correct++;
        else errors.push(`  "${examples[i].text}" : attendu ${Y[i] ? 'droite' : 'gauche'}, prédit ${Math.round((p >= 0.5 ? p : 1 - p) * 100)} % ${p >= 0.5 ? 'droite' : 'gauche'}`);
    }
}
console.log(`\nPrécision en validation croisée : ${((100 * correct) / examples.length).toFixed(1)} %`);
if (errors.length) console.log(`Exemples mal classés quand le modèle ne les a pas vus :\n${errors.join('\n')}`);

// Modèle final, entraîné sur tous les exemples.
const model = train(X, Y);
const round = (arr) => arr.map((v) => Math.round(v * 1e5) / 1e5);
writeFileSync(
    new URL('../site/classifier.json', import.meta.url),
    JSON.stringify({
        embeddingModel: EMBEDDING_MODEL,
        prefix: PREFIX,
        mean: round(model.mean),
        std: round(model.std),
        weights: round(model.weights),
        bias: model.bias,
        lookup: Object.fromEntries(examples.map((e) => [normalize(e.text), e.y])),
    }),
);
console.log('\nsite/classifier.json écrit.');
