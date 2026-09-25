import * as transformers from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/+esm';
import { classify, createEmbedder } from './classifier.js';

// Fait tourner le modèle dans un Web Worker : la page reste fluide pendant les calculs.
transformers.env.backends.onnx.wasm.proxy = true;

const EXAMPLES = ['la raclette', 'le télétravail', 'la trottinette électrique', 'le rugby', 'les chats', 'le camping-car', 'le yoga', 'la chasse au trésor'];

const $ = (id) => document.getElementById(id);
const form = $('form');
const input = $('q');
const go = $('go');
let ready = false;

// Tant que le modèle se télécharge, le formulaire est verrouillé. Sans ça, envoyer le
// formulaire recharge la page (envoi HTML classique) et relance le téléchargement à zéro.
function setBusy(value) {
    $('verdict').classList.toggle('busy', value);
    $('verdict').setAttribute('aria-busy', String(value));
}

function setReady(value) {
    ready = value;
    input.disabled = !value;
    go.disabled = !value;
    for (const chip of $('chips').children) chip.disabled = !value;
}

// Hémicycle : 5 rangées de sièges en demi-cercle, gauche en rouge, droite en bleu.
const seats = [];
for (let row = 0; row < 5; row++) {
    const radius = 44 + row * 11;
    const count = 9 + row * 3;
    for (let i = 0; i < count; i++) {
        const t = (i + 0.5) / count; // 0 = extrême gauche, 1 = extrême droite
        const angle = Math.PI * (1 - t);
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', (100 + radius * Math.cos(angle)).toFixed(2));
        c.setAttribute('cy', (100 - radius * Math.sin(angle)).toFixed(2));
        c.setAttribute('r', '3.4');
        c.setAttribute('class', `seat ${t < 0.5 ? 'seat-g' : 'seat-d'}`);
        seats.push({ el: c, t });
        $('seats').append(c);
    }
}

function showScore(pDroite) {
    // L'aiguille va de -90° (gauche) à +90° (droite).
    $('needle').style.transform = `rotate(${(pDroite - 0.5) * 180}deg)`;
    // On allume les sièges entre le centre et l'aiguille.
    for (const s of seats) {
        const on = pDroite >= 0.5 ? s.t >= 0.5 && s.t <= pDroite : s.t < 0.5 && s.t >= pDroite;
        s.el.classList.toggle('on', on);
    }
}

function verdict(pDroite) {
    const camp = pDroite >= 0.5 ? 'droite' : 'gauche';
    const confidence = pDroite >= 0.5 ? pDroite : 1 - pDroite;
    if (confidence < 0.55) return { camp: '', text: 'Ni l’un ni l’autre', confidence };
    if (confidence < 0.75) return { camp, text: `Plutôt de ${camp}`, confidence };
    return { camp, text: `C'est de ${camp}`, confidence };
}

for (const ex of EXAMPLES) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.textContent = ex;
    b.disabled = true;
    b.addEventListener('click', () => {
        input.value = ex;
        form.requestSubmit();
    });
    $('chips').append(b);
}

let classifier, embed;

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!ready || !text) return;
    setReady(false);
    setBusy(true);
    $('subject').textContent = `« ${text} »`;
    $('answer').textContent = 'Réflexion…';
    $('answer').className = 'answer';
    $('score').textContent = '';
    for (const s of seats) s.el.classList.remove('on');
    const start = performance.now();
    const { pDroite: p, source } = await classify(classifier, embed, text);
    const ms = performance.now() - start;
    const v = verdict(p);
    setBusy(false);
    showScore(p);
    $('answer').textContent = v.text;
    $('answer').className = `answer ${v.camp}`;
    $('score').textContent = `${Math.round((1 - p) * 100)} % gauche · ${Math.round(p * 100)} % droite`;
    $('status').textContent = source === 'exemple'
        ? 'Réponse tirée des exemples d’entraînement.'
        : `Deviné par le modèle en ${Math.round(ms)} ms, dans votre navigateur.`;
    setReady(true);
    input.focus();
});

// Le modèle est réparti en plusieurs fichiers (tokenizer, poids...) : on additionne leur progression.
const downloads = {};
function onProgress(e) {
    if (e.status !== 'progress' || !e.total) return;
    downloads[e.file] = e;
    const files = Object.values(downloads);
    const loaded = files.reduce((s, f) => s + f.loaded, 0);
    const total = files.reduce((s, f) => s + f.total, 0);
    $('progress').value = (100 * loaded) / total;
    if (loaded < total) {
        $('subject').textContent = `Téléchargement du modèle : ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} Mo`;
        $('answer').textContent = `${Math.floor((100 * loaded) / total)} %`;
    } else {
        // Téléchargé : le navigateur prépare encore le modèle pendant une seconde ou deux.
        $('subject').textContent = 'Préparation du modèle…';
        $('answer').textContent = '100 %';
    }
}

try {
    [classifier, embed] = await Promise.all([
        fetch('classifier.json').then((r) => r.json()),
        createEmbedder(transformers, { progress_callback: onProgress }),
    ]);
} catch (err) {
    setBusy(false);
    $('progress').hidden = true;
    $('subject').textContent = `Le modèle n'a pas pu être chargé (${err.message}).`;
    $('answer').textContent = 'Oups';
    $('score').textContent = 'Rechargez la page pour réessayer.';
    throw err;
}

$('progress').hidden = true;
setBusy(false);
$('status').textContent = 'Modèle prêt. Il tourne dans votre navigateur.';
$('subject').textContent = 'À vous de jouer.';
$('answer').textContent = '?';
$('score').textContent = '';
setReady(true);
input.focus();
