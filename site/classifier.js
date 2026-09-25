// Partagé entre le site (navigateur) et les scripts (Node).
import { EMBEDDING_MODEL, DTYPE, MODEL_FILE_NAME, toModelInput } from './config.js';

// Charge le modèle et renvoie une fonction qui transforme des textes en vecteurs.
// `transformers` est la librairie : importée depuis npm en Node, depuis le CDN dans le navigateur.
export async function createEmbedder({ AutoTokenizer, AutoModel }, options = {}) {
    const [tokenizer, model] = await Promise.all([
        AutoTokenizer.from_pretrained(EMBEDDING_MODEL, options),
        AutoModel.from_pretrained(EMBEDDING_MODEL, { dtype: DTYPE, model_file_name: MODEL_FILE_NAME, ...options }),
    ]);

    return async (texts) => {
        const inputs = await tokenizer(texts.map(toModelInput), { padding: true, truncation: true });
        const { sentence_embedding } = await model(inputs);
        return sentence_embedding.normalize(2, -1).tolist();
    };
}

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
export async function classify(classifier, embed, text) {
    // 1. Si le texte fait partie des exemples, on répond directement.
    const known = classifier.lookup[normalize(text)];
    if (known !== undefined) {
        return { pDroite: known, source: 'exemple' };
    }

    // 2. Sinon, le modèle transforme le texte en vecteur de 768 nombres...
    const [x] = await embed([text]);

    // 3. ... et la régression logistique en déduit la probabilité "droite".
    let z = classifier.bias;
    for (let i = 0; i < x.length; i++) {
        z += classifier.weights[i] * ((x[i] - classifier.mean[i]) / classifier.std[i]);
    }

    return { pDroite: 1 / (1 + Math.exp(-z)), source: 'modèle' };
}
