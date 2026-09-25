<?php

defined('CASTOR_USE_CHDIR') || define('CASTOR_USE_CHDIR', true);

use Castor\Attribute\AsArgument;
use Castor\Attribute\AsOption;
use Castor\Attribute\AsTask;

use function Castor\context;
use function Castor\io;
use function Castor\run;

#[AsTask(description: 'Installe les dépendances (transformers.js pour l\'entraînement)')]
function install(): void
{
    run('npm ci');
}

#[AsTask(description: 'Entraîne le classifieur à partir de data/exemples.csv et écrit site/classifier.json')]
function train(): void
{
    if (!is_dir(__DIR__ . '/node_modules')) {
        install();
    }

    run('node scripts/train.mjs', context: context()->withTimeout(null));
}

#[AsTask(description: 'Demande au classifieur si c\'est de gauche ou de droite, depuis le terminal')]
function predict(
    #[AsArgument(description: 'Un ou plusieurs sujets, ex. "le vélo" "facho"')]
    array $sujets,
): void {
    if (!is_file(__DIR__ . '/site/classifier.json')) {
        train();
    }

    run(['node', 'scripts/predict.mjs', ...$sujets]);
}

#[AsTask(description: 'Construit le site dans site/, prêt à être publié (utilisé par la CI)')]
function build(): void
{
    install();
    train();
}

#[AsTask(description: 'Sert le site en local pour le tester dans le navigateur')]
function serve(
    #[AsOption(description: 'Port d\'écoute')]
    int $port = 8000,
): void {
    if (!is_file(__DIR__ . '/site/classifier.json')) {
        train();
    }

    io()->success("Site disponible sur http://localhost:{$port} (Ctrl+C pour arrêter)");

    run(['php', '-S', "localhost:{$port}", '-t', 'site'], context: context()->withTimeout(null));
}
