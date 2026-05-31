# Projet Connect 5 - Groupe 6

## Contenu

- `group06.py` : agent `IntelligentPlayer` à remettre.
- `player.py` : base minimale utilisée par l'agent dans ce workspace.
- `PROMPTS_CONNECT5.md` : historique des prompts de travail.

## Idée de la solution

L'agent combine :

- détection immédiate des coups gagnants ;
- blocage immédiat des menaces adverses ;
- recherche minimax avec alpha-beta ;
- heuristique basée sur les alignements, le centre, et les menaces à deux issues.

## Fichier à remettre

Le livrable principal demandé par le sujet est :

- `group06.py`

La classe exportée est :

- `IntelligentPlayer`

## Test rapide

Si votre dossier de projet contient le moteur du jeu, vous pouvez tester avec :

```bash
python game.py --p1 player.HumanPlayer --p2 randomplayer.RandomPlayer
```

Puis remplacez `randomplayer.RandomPlayer` par votre agent.

## Vérification locale disponible dans ce workspace

Un test de syntaxe et un smoke test minimal ont déjà été exécutés sur l'agent, avec succès.

Si vous voulez refaire une vérification simple, vous pouvez lancer :

```bash
python -m py_compile player.py group06.py
```

## Remarque

Ce workspace ne contient pas le moteur complet du jeu. L'agent est donc écrit pour être robuste sur plusieurs formats de plateau, en attendant l'environnement d'évaluation fourni par l'enseignant.
