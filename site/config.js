// Partagé entre le script d'entraînement (Node) et le site (navigateur) :
// les deux doivent utiliser exactement le même modèle pour que les vecteurs soient comparables.
export const EMBEDDING_MODEL = 'Xenova/multilingual-e5-small';
export const PREFIX = 'query: '; // les modèles e5 attendent ce préfixe
export const DTYPE = 'q8'; // version quantifiée (~120 Mo au lieu de ~470 Mo)
