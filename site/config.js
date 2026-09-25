// Partagé entre le script d'entraînement (Node) et le site (navigateur) :
// les deux doivent utiliser exactement le même modèle pour que les vecteurs soient comparables.
export const EMBEDDING_MODEL = 'onnx-community/embeddinggemma-300m-ONNX';
export const DTYPE = 'q4'; // version quantifiée (~200 Mo au lieu de ~1,2 Go)
// Variante de la version q4 sans l'opération GatherBlockQuantized, que le moteur
// WebAssembly du navigateur ne sait pas exécuter.
export const MODEL_FILE_NAME = 'model_no_gather';

// EmbeddingGemma attend une consigne de tâche. Replacer le sujet dans un contexte
// politique fait aussi gagner quelques points de précision.
export const toModelInput = (text) => `task: classification | query: En politique française, ${text}`;
