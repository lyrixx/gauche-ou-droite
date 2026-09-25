// Partagé entre le site (navigateur) et les scripts (Node).
import { PREFIX } from './config.js';

// "La  Raclette !" et "raclette", ou "Jean-Marie Le Pen" et "jean marie le pen",
// doivent être reconnus comme le même exemple.
export function normalize(text) {
    return text
        .toLowerCase()
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '') // enlève les accents
        .replace(/[’`]/g, "'")
        .replace(/[-_]/g, ' ')
        .replace(/[.!?…]+$/g, '')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/^(?:de l'|l'|(?:les|le|la|l|un|une|des|du|de la|de l)\s+)/, ''); // enlève l'article
}

// Renvoie la probabilité "droite" (entre 0 et 1) et d'où vient la réponse.
export async function classify(classifier, extractor, text) {
    // 1. Si le texte fait partie des exemples, on répond directement.
    const known = classifier.lookup[normalize(text)];
    if (known !== undefined) {
        return { pDroite: known, source: 'exemple' };
    }

    // 2. Sinon, le modèle transforme le texte en vecteur de 384 nombres...
    const x = (await extractor(PREFIX + text, { pooling: 'mean', normalize: true })).data;

    // 3. ... et la régression logistique en déduit la probabilité "droite".
    let z = classifier.bias;
    for (let i = 0; i < x.length; i++) {
        z += classifier.weights[i] * ((x[i] - classifier.mean[i]) / classifier.std[i]);
    }

    return { pDroite: 1 / (1 + Math.exp(-z)), source: 'modèle' };
}
