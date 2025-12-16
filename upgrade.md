# Suivi des améliorations

## Terminé

- Gérer correctement les modèles ONNX : `backend/core/detectors/yolo.py` détecte désormais les exports ONNX et n’appelle plus `.to(device)` dessus.
- Reload de configuration fiable : suppression du cache, prise en compte dynamique de `CLV_CONFIG`, validation stricte (`grid_size`, `confidence`, `video_source`).
- Schéma/API/UX de config alignés : `ConfigSchema` bornée, champs supplémentaires (`inference_width`, `jpeg_quality`, `enable_backend_overlays`) exposés dans l’UI.
- Superposition vidéo recalibrée : canvas dimensionné sur la taille affichée, échelle calculée à partir de `frame_size`, élimination du drift.
- WebSocket robuste : reconnexion avec backoff et normalisation `ws://`/`wss://`.
- Signalisation d’erreur pipeline : `/stats` retourne `error` depuis le moteur, `VideoEngine` enregistre les échecs de source/processing.
- Nettoyage frontend : dépendances inutilisées retirées, fichiers Vite dupliqués/artefacts ignorés via `.gitignore`.
- CLI vision : `run_on_video` expose `grid_size`, `smoothing`, `inference_width` et respecte `_parse_grid`.
- UX header/fps : horloge actualisée, compteur FPS/people rafraîchi grâce au flux WS résilient.
- Tests ajoutés : couverture pour settings (reload + validation), ONNX detector (pas de `.to`), endpoints FastAPI (`/health`, `/stats`), échec source vidéo, validation `/config`, exécution CLI `run_on_video` sur vidéo factice.
- Observabilité UI : affichage erreurs backend dans la sidebar et bandeau vidéo, état buffering explicite.
- Docker optimisé : multi-stage builder/runtime, dépendances minimalistes, utilisateur non-root, restart policy et shm dans compose (CPU-only).

## À faire

1) Optionnel: exécuter `python scripts/dev/prune_videos.py --apply` pour ne conserver que 3 vidéos d’exemple (les tests n’en dépendent pas).
