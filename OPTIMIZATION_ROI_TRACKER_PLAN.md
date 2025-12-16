# Optimisation CPU avancée : ROI dynamique pilotée par le tracker (+ détection périphérique)

## Objectif

Réduire drastiquement le temps d’inférence YOLO (CPU-only) sans casser :

- la **détection multi-personnes**,
- le **tracking**,
- les **analytics** (densité/compteurs),
- les **targets head/body lock-on**.

L’idée : **ne plus faire d’inférence plein cadre à chaque frame**. À la place, on n’infère que sur des **crops** (ROI) à forte probabilité de contenir des personnes, dérivées des tracks existants, + une petite zone “entrée” pour capter les nouveaux arrivants.

## Principe (résumé)

À chaque frame :

1. **ROI “tracks” (priorité haute)** : pour chaque track actif, utiliser la bbox prédite au frame t+1 et l’élargir (marge) pour absorber mouvement + erreurs.
2. **ROI “entrée” (périphérie)** : ajouter des bandes fines (bords gauche/droite/bas typiquement) pour capter les nouveaux entrants sans repasser tout le frame.
3. **Fallback plein cadre** : toutes les N frames (ou sur signaux de dégradation), forcer une détection plein cadre pour corriger la dérive et récupérer les cas difficiles.

Ensuite :

- on **reprojette** les détections des ROI vers le repère plein cadre,
- on **fusionne** (NMS/merge) si besoin,
- puis on alimente le **tracker** et l’analytics comme aujourd’hui.

## Pourquoi ça colle au code CrowdLock Vision

Le repo est explicitement : détection → tracking → analytics/targets, en temps réel.

- Le tracker fournit exactement l’information la plus précieuse pour accélérer : **où chercher au frame suivant**.
- Les targets head/body sont sensibles à la stabilité : tracking-first + ré-inférence localisée améliore souvent la stabilité perçue.
- CPU-only : le coût dominant est l’inférence sur des images larges. Réduire la surface traitée est un levier majeur.

## Design technique (détails)

### A) Génération des ROI

1) **ROI par track**

   - Entrée : liste de tracks actifs (bbox + état + confiance).
   - Sortie : liste de rectangles ROI en coordonnées image.
   - Expansion :
     - marge relative (ex: +30% largeur/hauteur),
     - plus grande marge si : vitesse élevée, track instable, caméra mobile détectée.

2) **ROI périphérie (bande d’entrée)**

   - Exemples simples :
     - bande bas : `y in [H*(1-p), H]`
     - bande gauche/droite : `x in [0, W*p]` et `x in [W*(1-p), W]`
   - Raison : beaucoup de nouveaux entrants arrivent par les bords.

3) **Fusion des ROI avant inférence**

   - Si plusieurs ROI se chevauchent, les fusionner (union) pour :
     - réduire le nombre de passages modèle,
     - éviter les doublons de détection.

4) **Cap budget ROI**

   - Éviter le pire cas “scène très dense” où les ROI couvrent quasi 100%.
   - Stratégie :
     - si aire(ROI_total) > seuil (ex: 70%), alors **plein cadre** directement.

### B) Inférence sur ROI

- Pour chaque ROI fusionnée :
  - crop image,
  - inférence YOLO,
  - reprojection des boîtes vers le plein cadre.

### C) Fusion & suppression des doublons

- Les détections issues de multiples ROI peuvent se recouper.
- Appliquer un NMS/merge en coordonnées plein cadre.
- Utiliser un IoU threshold cohérent avec l’existant.

### D) Stratégie de fallback (plein cadre)

Déclenchement par :

- **période** : toutes les N frames (ex: 10–30 selon perf),
- **signal tracker** :
  - % de tracks perdus soudainement,
  - forte hausse d’incertitude/variance,
  - mismatch persistant (associations qui échouent),
- **signal “camera motion”** (optionnel) :
  - mesure cheap de changement global (diff d’histogramme, flow approximé, etc.).

## Paramètres recommandés (premier jet)

À ajuster via config (YAML/env) pour permettre tuning rapide :

- `roi_enabled`: bool
- `roi_track_margin`: float (ex: 0.30)
- `roi_entry_band`: float (ex: 0.08)  # fraction largeur/hauteur
- `roi_merge_iou`: float (ex: 0.10 à 0.30 pour fusion rectangles)
- `roi_max_area_fraction`: float (ex: 0.70)
- `roi_full_frame_every_n`: int (ex: 15)
- `roi_force_full_frame_on_track_loss`: float (ex: 0.25)  # si >25% de tracks drop

## Risques & mitigations

- **Risque : rater un nouvel entrant au centre** (pas par les bords).
  - Mitigation : fallback périodique + (option) petite ROI “motion hotspot”.
- **Risque : dérive si caméra mobile**.
  - Mitigation : augmenter marges + fallback plus fréquent lorsque mouvement global.
- **Risque : doublons / NMS incorrect**.
  - Mitigation : fusion en plein cadre systématique, tests unitaires sur merge.
- **Risque : ROI trop petites → baisse recall**.
  - Mitigation : marges adaptatives selon vitesse/incertitude.

## Plan d’implémentation (proposé)

### Étape 1 — Cadrage dans l’architecture actuelle

- Localiser le point unique où l’on appelle le detector YOLO (pipeline/engine).
- Introduire un mode `roi_enabled` qui choisit :
  - détection plein cadre (actuel),
  - détection ROI (nouveau),
  - fallback plein cadre selon règles.

Livrable : design clair + emplacement du changement (un seul point si possible).

### Étape 2 — Implémenter la génération et fusion de ROI

- Fonction pure : `build_rois(tracks, frame_w, frame_h, cfg) -> list[ROI]`.
- Fonction pure : `merge_rois(rois, iou_threshold) -> list[ROI]`.
- Logique “cap area” : si trop grand → plein cadre.

Livrable : code testable sans modèle.

### Étape 3 — Exécuter l’inférence sur ROI + reprojection

- Adapter la fonction d’inférence :
  - itère sur ROI,
  - infère sur crop,
  - reprojette les boxes.
- Fusion/NMS des détections en plein cadre.

Livrable : pipeline fonctionnel en ROI-only.

### Étape 4 — Ajouter fallback périodique + triggers tracker

- Implémenter `full_frame_every_n`.
- Trigger sur pertes de tracks.
- (Optionnel) trigger “camera motion”.

Livrable : robustesse sur scènes dures.

### Étape 5 — Bench & validation

- Utiliser `backend.tools.bench_video` pour comparer :
  - `roi_enabled=false` vs `true`.
- Mesurer :
  - FPS,
  - latence “detect” (p50/p90/p99),
  - stabilité des targets (moins de jitter),
  - taux de tracks perdus.

Critère de succès (indicatif) :

- gain FPS/latence sur scènes calmes/modérées,
- pas de dégradation majeure sur densité élevée (fallback prend le relais).

## Resultat

Benchmark exécuté sur :

- vidéo : `testdata/videos/855564-hd_1920_1080_24fps.mp4`
- modèle : `yolov8n.pt` (task `detect`)
- paramètres communs : `inference_width=640`, `inference_stride=1`, `warmup_frames=20`, `max_frames=150`

Commandes :

- sans ROI : `python -m backend.tools.bench_video --input testdata/videos/855564-hd_1920_1080_24fps.mp4 --model yolov8n.pt --task detect --inference-width 640 --inference-stride 1 --warmup-frames 20 --max-frames 150 --out benchmark_baseline_no_roi.json`
- sans ROI : `python -m backend.tools.bench_video --input testdata/videos/855564-hd_1920_1080_24fps.mp4 --model yolov8n.pt --task detect --inference-width 640 --inference-stride 1 --warmup-frames 20 --max-frames 150 --out benchmark_baseline_no_roi_v2.json`
- avec ROI : `python -m backend.tools.bench_video --input testdata/videos/855564-hd_1920_1080_24fps.mp4 --model yolov8n.pt --task detect --inference-width 640 --inference-stride 1 --roi --roi-track-margin 0.30 --roi-entry-band 0.08 --roi-merge-iou 0.20 --roi-max-area-fraction 0.70 --roi-full-frame-every-n 15 --roi-force-full-frame-on-track-loss 0.25 --roi-detections-nms-iou 0.50 --warmup-frames 20 --max-frames 150 --out benchmark_roi_enabled_v2.json`

Tableau (ms = millisecondes, plus bas = mieux) :

| Mode | FPS | Detect mean (ms) | Detect p50 (ms) | Detect p95 (ms) | Pipeline mean (ms) | Total mean (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Sans ROI | 10.45 | 82.39 | 55.71 | 200.68 | 83.35 | 95.70 |
| ROI activé | 15.98 | 51.24 | 49.71 | 54.45 | 52.13 | 62.57 |
| Δ (ROI - sans ROI) | +5.53 | -31.15 | -6.00 | -146.23 | -31.22 | -33.12 |

Notes :

- Ici `inference_stride=1` ⇒ 100% des frames mesurées font une inférence (pas de frames “skipped”).
- Interprétation rapide : sur ce run précis, le gain est net (moins de surface inférée + fusion ROI → moins d’overhead), avec une chute très forte de la latence p95.

### Étape 6 — Tests

Ajouter des tests unitaires (sans gros modèle) :

- ROI expand/clamp dans l’image,
- merge ROI,
- reprojection des boxes,
- logique de fallback.

## Notes pratiques

- Sur CPU, le gain dépend surtout de :
  - % de surface réellement inférée,
  - nombre d’appels au modèle (fusion ROI importante),
  - coût resize/normalize.
- Une implémentation simple (marges fixes + bandes) est déjà très rentable.

---

### Checklist rapide (tuning)

- Si vous ratez des entrants : augmenter `roi_entry_band` ou baisser `roi_full_frame_every_n`.
- Si vous perdez des tracks : augmenter `roi_track_margin` ou déclencher fallback plus tôt.
- Si la scène est dense : `roi_max_area_fraction` force souvent plein cadre (comportement attendu).
