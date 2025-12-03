# Améliorations prioritaires

1) [Critique] Nettoyer le dépôt : retirer les poids YOLO et exports embarqués (`yolov8n.pt`, `yolov8n-pose.pt`, `yolov8n.onnx`) ainsi que les artefacts de build front (`web/tsconfig*.tsbuildinfo`, `web/vite.config.d.ts`). Conserver seulement un jeu minimal de vidéos d’exemple (3 clips couvrant les cas du SPEC) ou fournir un script de téléchargement pour les autres fichiers volumineux.

2) [Critique] Gérer correctement les modèles ONNX : `backend/core/detectors/yolo.py` appelle `.to(device)` même sur un modèle ONNX, ce qui déclenche l’erreur capturée dans `benchmark_error.txt`. Détecter le format et utiliser le runtime approprié (ONNXRuntime / pas de `.to`), sinon refuser le device.

3) [Critique] Reload de configuration cassé : `load_settings()` est mémoïsée (`lru_cache`) et `reload_settings()` reconstruit les réglages à partir du cache. Toute modification du fichier `config/backend.config.yml` ou des variables d’environnement est donc ignorée après le premier chargement. Supprimer le cache ou l’invalider dans `reload_settings`, et valider les champs (notamment `grid_size`).

4) [Élevée] Schéma de config incomplet/incohérent : `ConfigSchema` n’impose ni format ni bornes (`grid_size` libre, `confidence` non bornée). L’UI (`web/src/components/SourceSelector.tsx`) ne permet pas d’éditer `grid_size`, `smoothing`, `device`, ou `video_path`/`rtsp_url` quand le type change, et force des valeurs par défaut différentes du backend (UI `smoothing=0.2` vs backend `0.9`). Aligner les défauts, ajouter validation (regex `^\d+x\d+$`, bornes 0‑1 pour confidence) et exposer tous les champs.

5) [Élevée] Superposition vidéo décalée : `web/src/components/VideoOverlay.tsx` dessine sur un canvas 1280×720 fixe sans tenir compte de la taille réelle du flux (ni de `frame_size`). Les boxes/heatmaps sont fausses dès que la vidéo n’a pas exactement ce ratio. Dimensionner le canvas à la taille naturelle de l’image (ou au `frame_size` reçu) et le recalibrer sur `resize`.

6) [Élevée] Robustesse websocket : `web/src/hooks/useWebsocket.ts` n’a ni reconnexion ni sélection `ws://`/`wss://`. Une coupure réseau stoppe définitivement le flux et le statut reste “closed”. Ajouter backoff, heartbeat/ping et détection du schéma à partir de `VITE_API_BASE`.

7) [Élevée] Signalisation d’échec des sources vidéo : `VideoEngine.start()` avale les exceptions d’ouverture de source (File/RTSP/Webcam) et `/stats` continue à répondre avec des zéros. Exposer un état d’erreur (health détaillé ou champ dans `/stats`) et logguer l’échec pour que l’UI puisse l’afficher.

8) [Moyenne] Optimiser les images Docker : backend Dockerfile monolithique (root, pas de multi‑stage, torch CPU forcé) et `docker-compose.yml` sans options GPU ou `restart`. Introduire build multi‑stage (builder + runtime slim, utilisateur non‑root), ARG pour choisir CPU/GPU, purge des caches pip, et flags GPU/`restart: unless-stopped` dans compose.

9) [Moyenne] Couverture de tests insuffisante : seuls les calculs analytics sont testés. Ajouter des tests FastAPI (health/config/stats, flux WS simulé), tests de reload settings, tests d’échec de sources vidéo, et un test CLI `run_on_video` sur une courte vidéo factice afin d’atteindre l’objectif de couverture (>80%).

10) [Moyenne] Dépendances et fichiers doublons côté web : `@tanstack/react-query` et `zustand` ne sont pas utilisés, deux configs Vite cohabitent (`vite.config.ts` et `.js`) et des artefacts de compilation sont versionnés. Nettoyer les dépendances, conserver un seul fichier Vite et ajouter `.gitignore` pour les artefacts.

11) [Moyenne] CLI et benchmark à parfaire : `backend/tools/run_on_video.py` n’expose pas `inference_width`, `grid_size` ou `smoothing`, produisant des JSON volumineux et difficiles à comparer. `backend/benchmark_vision.py` doit utiliser le chemin ONNX sans `.to(device)` et permettre de fixer le backend (cpu/cuda) et les paramètres de pipeline pour des mesures reproductibles.

12) [Moyenne] UX/observabilité : le header affiche une date statique, l’UI ne montre pas d’état “buffering” ou d’erreur sur le flux vidéo, et les toggles n’agissent que côté front (pas d’option pour désactiver l’overlay serveur et économiser la bande passante). Ajouter états de chargement/erreur, synchroniser les toggles avec un paramètre backend `enable_backend_overlays`, et afficher des métriques clés (latence, taille de frame, densest cell lisible).
