# Tiny Object Detection Research Exploration

## Scope and decision rule
This report is a diagnostic exploration, not a method claim. A hypothesis is promoted only when the same directional signal is observed on at least two of LEVIR-Ship, VisDrone2019-DET, and TinyPerson. The probes are lightweight proxies and do not modify detector behavior.

## Existing baseline context
The project baseline reports show a consistent tiny-object difficulty gap: standard YOLOv8n test `mAP50-95` is approximately 0.264 on LEVIR-Ship, 0.151 on VisDrone mosaic, and 0.178 on TinyPerson. These are context values, not probe outcomes, and their split-qualified provenance remains in the source reports.

## Experiments performed
1. Causal intervention: paired original, object-removed, object-only, and context-only image inference. `background_dependence` is context-only score divided by original score, while `object_sensitivity` is the original minus object-removed score.
2. Information bottleneck: hook-level object-region activation fraction and context nuisance proxy across available backbone/P2/P3/P4/P5 modules.
3. Object survival: object-region energy relative to surrounding context energy across stages.
4. Gradient optimization: inference-only activation-objective gradient norm, variance, and signal-to-noise proxy. This is a diagnostic of feature sensitivity, not a replacement for full training gradient accounting.
5. Decisive follow-up: raw-candidate local-pool analysis with the report-listed dataset-specific YOLOv8 checkpoints. For every GT object, candidates are restricted to a padded local neighborhood, then the highest-IoU candidate is compared with the score-selected candidate.
6. Training-feasibility gate: brightness and blur perturbations test candidate identity stability, while a frozen train/test calibration tests whether score, feature energy, and box geometry can learn a better utility ranking without changing the detector.

Artifacts available for: levir-ship, visdrone, tinyperson.

## Dataset-level evidence
### levir-ship

- Causal rows: 19. Size-conditioned object/background summaries: `{"medium_ge32": {"background_dependence": 5.346724996478653, "object_sensitivity": 0.0640983300593992, "original_score": 0.0977516674126188}, "small_16_32": {"background_dependence": 2.919308374745997, "object_sensitivity": 0.1247631254055622, "original_score": 0.19798246601765807}, "tiny_8_16": {"background_dependence": 67.84764639272467, "object_sensitivity": 0.0019505565287545323, "original_score": 0.0019505565287545323}}`.
- Feature information summaries: `{"model.12": {"nuisance_proxy": 0.9997445312989397, "object_fraction": 1.1208440465699876}, "model.15": {"nuisance_proxy": 0.9995202806972784, "object_fraction": 1.2541416009326876}, "model.18": {"nuisance_proxy": 0.9991216043713197, "object_fraction": 1.4318987853536975}, "model.21": {"nuisance_proxy": 0.9995069863212853, "object_fraction": 1.1967143428567926}, "model.4": {"nuisance_proxy": 0.9992888938387482, "object_fraction": 1.451880228708694}, "model.8": {"nuisance_proxy": 1.000064966466896, "object_fraction": 0.9740764377150443}}`.
- Survival curves: `{"model.12": {"medium_ge32": 1.1290643238033324, "small_16_32": 1.157102500219007, "tiny_8_16": 0.8996599158557937}, "model.15": {"medium_ge32": 1.2518253928899792, "small_16_32": 1.2988187636178206, "tiny_8_16": 1.0213075350704492}, "model.18": {"medium_ge32": 1.4361432738291013, "small_16_32": 1.5178361043142004, "tiny_8_16": 0.959861044354499}, "model.21": {"medium_ge32": 1.4028892782983158, "small_16_32": 1.1260934434293186, "tiny_8_16": 0.9745042682438434}, "model.4": {"medium_ge32": 1.3720643664023147, "small_16_32": 1.4798320731991337, "tiny_8_16": 1.5473749052717545}, "model.8": {"medium_ge32": 0.9167702580796604, "small_16_32": 1.0041106420578747, "tiny_8_16": 0.9803486843955016}}`.
- Gradient summaries: `{"model.12": {"gradient_norm": 0.05842533295876101, "gradient_snr": 0.005114526592047983, "gradient_std": 0.0001291010109860891}, "model.15": {"gradient_norm": 0.04659027959171094, "gradient_snr": 0.008753798951051737, "gradient_std": 7.279485649441515e-05}, "model.18": {"gradient_norm": 0.05842533295876101, "gradient_snr": 0.006286881232977305, "gradient_std": 0.00012909990657231232}, "model.21": {"gradient_norm": 0.0625, "gradient_snr": 0.027437077688151283, "gradient_std": 0.00019523376173119207}, "model.4": {"gradient_norm": 0.04659027959171094, "gradient_snr": 0.004779349265031908, "gradient_std": 7.279639401914258e-05}, "model.8": {"gradient_norm": 0.0625, "gradient_snr": 0.022146846020692272, "gradient_std": 0.00019526379918189426}}`.
- Matched candidate evidence: `{"medium_ge32": {"evidence_iou": 0.5932758814758725, "evidence_rank_corr_iou": 0.7385190995616742, "oracle_gap": 0.08054365714391072, "oracle_iou": 0.8055167198181152, "score_iou": 0.7249730626742045, "score_rank_corr_iou": 0.8885727501018358}, "small_16_32": {"evidence_iou": 0.6623698132378715, "evidence_rank_corr_iou": 0.7487648587510592, "oracle_gap": 0.092924359866551, "oracle_iou": 0.7489885815552303, "score_iou": 0.6560642216886793, "score_rank_corr_iou": 0.8657744744523367}, "tiny_8_16": {"evidence_iou": 0.3785139146176251, "evidence_rank_corr_iou": 0.6748738960391157, "oracle_gap": 0.059465607458894905, "oracle_iou": 0.4413518946279179, "score_iou": 0.381886287169023, "score_rank_corr_iou": 0.8153432122708552}}`.
- Alignment feasibility: `{"calibrated_corr": 0.6881999849360917, "calibrated_mse": 0.028596955242049953, "calibrated_top_iou": 0.6247456171191655, "oracle_top_iou": 0.7183106455665368, "raw_mse": 0.13549634580728775, "raw_score_corr": 0.7674516018454999, "raw_top_iou": 0.6384600693216691}` and stability `{"medium_ge32": {"aug_oracle_iou": 0.8002911761954978, "base_oracle_iou": 0.8055167198181152, "oracle_identity_stable": 0.7407407407407407, "score_identity_stable": 0.9259259259259259, "score_rank_corr": 0.9946550814381568}, "small_16_32": {"aug_oracle_iou": 0.7433990995089214, "base_oracle_iou": 0.7489885815552303, "oracle_identity_stable": 0.7238095238095238, "score_identity_stable": 0.7714285714285715, "score_rank_corr": 0.9924693813613034}, "tiny_8_16": {"aug_oracle_iou": 0.43895225994514697, "base_oracle_iou": 0.4413518946279179, "oracle_identity_stable": 0.7878787878787878, "score_identity_stable": 0.8181818181818182, "score_rank_corr": 0.975788701988452}}`.

### visdrone

- Causal rows: 704. Size-conditioned object/background summaries: `{"medium_ge32": {"background_dependence": 0.47646818094754506, "object_sensitivity": 0.5380913861821183, "original_score": 0.7205998961904407}, "small_16_32": {"background_dependence": 0.3291365609200414, "object_sensitivity": 0.5752243198910538, "original_score": 0.7436073890754155}, "tiny_8_16": {"background_dependence": 0.3551397995389954, "object_sensitivity": 0.638067890322063, "original_score": 0.7339397508923601}}`.
- Feature information summaries: `{"model.12": {"nuisance_proxy": 0.9991528956283321, "object_fraction": 1.1198927619933934}, "model.15": {"nuisance_proxy": 0.9990691130325131, "object_fraction": 1.2102835635091396}, "model.18": {"nuisance_proxy": 0.9985862791823839, "object_fraction": 1.2637694779119515}, "model.21": {"nuisance_proxy": 0.9992067673157521, "object_fraction": 1.0439316514121504}, "model.4": {"nuisance_proxy": 0.9993180540510603, "object_fraction": 1.1486598604149003}, "model.8": {"nuisance_proxy": 0.9998374094085275, "object_fraction": 1.020677944371169}}`.
- Survival curves: `{"model.12": {"medium_ge32": 1.1286416301815883, "small_16_32": 1.084807701154569, "tiny_8_16": 1.1000730058426964}, "model.15": {"medium_ge32": 1.2133141329804962, "small_16_32": 1.182114068645896, "tiny_8_16": 1.255121368297782}, "model.18": {"medium_ge32": 1.273203235295328, "small_16_32": 1.2202908083341182, "tiny_8_16": 1.2701527699854829}, "model.21": {"medium_ge32": 1.0619384279816872, "small_16_32": 0.9920844086524625, "tiny_8_16": 0.9383074069444723}, "model.4": {"medium_ge32": 1.142785458436404, "small_16_32": 1.1596240852418265, "tiny_8_16": 1.2169496342373287}, "model.8": {"medium_ge32": 1.0197576171734468, "small_16_32": 1.0148896634015494, "tiny_8_16": 1.0508787226045087}}`.
- Gradient summaries: `{"model.12": {"gradient_norm": 0.048392486466168935, "gradient_snr": 0.01123879990764594, "gradient_std": 0.00010692767328893379}, "model.15": {"gradient_norm": 0.039344801854564466, "gradient_snr": 0.021634012307582783, "gradient_std": 6.146731491376364e-05}, "model.18": {"gradient_norm": 0.048392486466168935, "gradient_snr": 0.012433332347198867, "gradient_std": 0.00010692673290112576}, "model.21": {"gradient_norm": 0.05007718033316037, "gradient_snr": 0.041630988986898126, "gradient_std": 0.0001563752103844639}, "model.4": {"gradient_norm": 0.039344801854564466, "gradient_snr": 0.009295113643206192, "gradient_std": 6.147431698105328e-05}, "model.8": {"gradient_norm": 0.05007718033316037, "gradient_snr": 0.03339794901231388, "gradient_std": 0.00015641812063646202}}`.
- Matched candidate evidence: `{"medium_ge32": {"evidence_iou": 0.3500494571946564, "evidence_rank_corr_iou": 0.15294100303674044, "oracle_gap": 0.3137784337211652, "oracle_iou": 0.7317362932530247, "score_iou": 0.417957860032932, "score_rank_corr_iou": 0.2935106909209511}, "small_16_32": {"evidence_iou": 0.3293562123179898, "evidence_rank_corr_iou": 0.16925789535085928, "oracle_gap": 0.2762892243288291, "oracle_iou": 0.6178166010687428, "score_iou": 0.34152737680722084, "score_rank_corr_iou": 0.3163127679813616}, "tiny_8_16": {"evidence_iou": 0.2679848815374947, "evidence_rank_corr_iou": 0.1653913272066951, "oracle_gap": 0.2493413934459934, "oracle_iou": 0.48012966652969263, "score_iou": 0.23078827278132175, "score_rank_corr_iou": 0.36750805717677176}, "tiny_lt8": {"evidence_iou": 0.05703354673460126, "evidence_rank_corr_iou": 0.01712454212454211, "oracle_gap": 0.2963958643376827, "oracle_iou": 0.30143187567591667, "score_iou": 0.005036011803895235, "score_rank_corr_iou": -0.005731768231768257}}`.
- Alignment feasibility: `{"calibrated_corr": 0.35860213588846923, "calibrated_mse": 0.04788331151186282, "calibrated_top_iou": 0.3933329565815436, "oracle_top_iou": 0.707489961958125, "raw_mse": 0.1039265923885302, "raw_score_corr": 0.20808933636317237, "raw_top_iou": 0.3939735332173879}` and stability `{"medium_ge32": {"aug_oracle_iou": 0.7362392160222972, "base_oracle_iou": 0.7317362932530247, "oracle_identity_stable": 0.6490889603429796, "score_identity_stable": 0.7311897106109325, "score_rank_corr": 0.9745885169343766}, "small_16_32": {"aug_oracle_iou": 0.6170439934612207, "base_oracle_iou": 0.6178166010687428, "oracle_identity_stable": 0.760132340777502, "score_identity_stable": 0.7857733664185277, "score_rank_corr": 0.972394574497598}, "tiny_8_16": {"aug_oracle_iou": 0.4760966376115233, "base_oracle_iou": 0.48012966652969263, "oracle_identity_stable": 0.8528138528138528, "score_identity_stable": 0.8484848484848485, "score_rank_corr": 0.9756339575540027}, "tiny_lt8": {"aug_oracle_iou": 0.2736536941180627, "base_oracle_iou": 0.30143187567591667, "oracle_identity_stable": 0.9166666666666666, "score_identity_stable": 0.6666666666666666, "score_rank_corr": 0.9215839715839715}}`.

### tinyperson

- Causal rows: 100. Size-conditioned object/background summaries: `{"medium_ge32": {"background_dependence": 0.47320837036157465, "object_sensitivity": 0.35135417043541867, "original_score": 0.4753046086989343}, "small_16_32": {"background_dependence": 0.7369701908457283, "object_sensitivity": 0.15865750938907944, "original_score": 0.38641591742634773}, "tiny_8_16": {"background_dependence": 1.0564583099782487, "object_sensitivity": 0.00648188591003418, "original_score": 0.3860059976577759}, "tiny_lt8": {"background_dependence": 1.0564583099782492, "object_sensitivity": 0.00648188591003418, "original_score": 0.3860059976577759}}`.
- Feature information summaries: `{"model.12": {"nuisance_proxy": 0.9995349217820143, "object_fraction": 1.2031900125188202}, "model.15": {"nuisance_proxy": 0.9995399497461964, "object_fraction": 1.3496589572938462}, "model.18": {"nuisance_proxy": 0.9990459072891079, "object_fraction": 1.5156195391140945}, "model.21": {"nuisance_proxy": 0.9993264282238883, "object_fraction": 1.1076991777410408}, "model.4": {"nuisance_proxy": 0.9995481307685088, "object_fraction": 1.3654059016954592}, "model.8": {"nuisance_proxy": 0.9997966632718089, "object_fraction": 1.041659164876475}}`.
- Survival curves: `{"model.12": {"medium_ge32": 1.2380163291482111, "small_16_32": 1.1971219930268602, "tiny_8_16": 1.194719552290839, "tiny_lt8": 1.2411668135695195}, "model.15": {"medium_ge32": 1.2573327163104946, "small_16_32": 1.427973728827643, "tiny_8_16": 1.3196859177604896, "tiny_lt8": 1.1599310505941551}, "model.18": {"medium_ge32": 1.52896495794491, "small_16_32": 1.8040626565460074, "tiny_8_16": 1.2380740587756176, "tiny_lt8": 1.159031301393689}, "model.21": {"medium_ge32": 1.3444897820336241, "small_16_32": 1.1473133344146806, "tiny_8_16": 1.0009455552798352, "tiny_lt8": 1.0358235974369137}, "model.4": {"medium_ge32": 1.223484160942236, "small_16_32": 1.4100451020011435, "tiny_8_16": 1.405874403741846, "tiny_lt8": 1.0755407863172601}, "model.8": {"medium_ge32": 1.0846842492170927, "small_16_32": 1.0150607980207538, "tiny_8_16": 1.0659700243561778, "tiny_lt8": 1.000657887713538}}`.
- Gradient summaries: `{"model.12": {"gradient_norm": 0.07413914041593671, "gradient_snr": 0.0075105345458723605, "gradient_std": 0.00016382114503358025}, "model.15": {"gradient_norm": 0.07562390227802097, "gradient_snr": 0.009638714105822146, "gradient_std": 0.0001181585438826005}, "model.18": {"gradient_norm": 0.07413914041593671, "gradient_snr": 0.00572834703954868, "gradient_std": 0.0001638227212970378}, "model.21": {"gradient_norm": 0.05911711074411869, "gradient_snr": 0.03063428514637053, "gradient_std": 0.00018465673783794046}, "model.4": {"gradient_norm": 0.07562390227802097, "gradient_snr": 0.005366767593077384, "gradient_std": 0.0001181609875857248}, "model.8": {"gradient_norm": 0.05911711074411869, "gradient_snr": 0.02454743139445782, "gradient_std": 0.00018468876951374113}}`.
- Matched candidate evidence: `{"medium_ge32": {"evidence_iou": 0.4911051234778236, "evidence_rank_corr_iou": 0.29449545004353356, "oracle_gap": 0.29890883582479816, "oracle_iou": 0.8035741110934931, "score_iou": 0.5046652759260991, "score_rank_corr_iou": 0.5751756101596187}, "small_16_32": {"evidence_iou": 0.42615972596220675, "evidence_rank_corr_iou": 0.35336116705719406, "oracle_gap": 0.24305841076374055, "oracle_iou": 0.6892187593579292, "score_iou": 0.44616034906357527, "score_rank_corr_iou": 0.4709270504324929}, "tiny_8_16": {"evidence_iou": 0.299916011372076, "evidence_rank_corr_iou": 0.21000846856663846, "oracle_gap": 0.3194808713712935, "oracle_iou": 0.6048051993978225, "score_iou": 0.2853243284645664, "score_rank_corr_iou": 0.29314686949952207}, "tiny_lt8": {"evidence_iou": 0.2728817860285441, "evidence_rank_corr_iou": 0.020848595848595837, "oracle_gap": 0.2722114556365543, "oracle_iou": 0.5083131161000993, "score_iou": 0.23610166046354505, "score_rank_corr_iou": 0.16474821474821477}}`.
- Alignment feasibility: `{"calibrated_corr": 0.3213989151337761, "calibrated_mse": 0.04462140990536389, "calibrated_top_iou": 0.3652142297183648, "oracle_top_iou": 0.6657600015132136, "raw_mse": 0.08981101979383556, "raw_score_corr": 0.31155016219276915, "raw_top_iou": 0.38079984768239017}` and stability `{"medium_ge32": {"aug_oracle_iou": 0.7982333721775635, "base_oracle_iou": 0.8035741110934931, "oracle_identity_stable": 0.5931372549019608, "score_identity_stable": 0.8088235294117647, "score_rank_corr": 0.9937732085398384}, "small_16_32": {"aug_oracle_iou": 0.6858648567795753, "base_oracle_iou": 0.6892187593579292, "oracle_identity_stable": 0.7893333333333333, "score_identity_stable": 0.8666666666666667, "score_rank_corr": 0.9900358745189556}, "tiny_8_16": {"aug_oracle_iou": 0.5972861875964086, "base_oracle_iou": 0.6048051993978225, "oracle_identity_stable": 0.8870056497175142, "score_identity_stable": 0.884180790960452, "score_rank_corr": 0.9830632646247472}, "tiny_lt8": {"aug_oracle_iou": 0.4961014853583442, "base_oracle_iou": 0.5083131161000993, "oracle_identity_stable": 0.9259259259259259, "score_identity_stable": 0.8518518518518519, "score_rank_corr": 0.9784249700916369}}`.

## Hypothesis assessment

### 1. Causal feature intervention: reject as the primary direction
The matched checkpoints do not show one consistent background-shortcut pattern. VisDrone tiny objects retain substantial context-only score, while TinyPerson tiny objects are highly sensitive to context removal, and LEVIR uses a different score regime. This is useful as a failure analysis, but not a stable cross-dataset method hypothesis.

### 2. Tiny object information bottleneck: reject the simple compression story
Object-region activation remains measurable in intermediate and late features. The feature curves are non-monotonic rather than progressively collapsing, and the activation proxy is not a held-out information estimator. The evidence does not justify a generic information-preserving module.

### 3. Object survival modeling: reject as a universal law, retain as a measurement
The survival score is useful for locating weak stages, but it does not decrease monotonically across LEVIR-Ship, VisDrone, and TinyPerson. It should remain an evaluation signal rather than become the method itself.

### 4. Gradient optimization: reject as the first intervention
The activation-gradient proxy varies by layer, but it does not measure true classification, box, or DFL training gradients. There is no cross-dataset causal evidence that gradient instability is the dominant failure. Do not begin with a new loss.

### 5. Candidate evidence-to-score misalignment: supported and actionable
The decisive matched-checkpoint result is a large local oracle gap on VisDrone and TinyPerson. The score-selected candidate trails the best-IoU candidate by roughly 0.25--0.32 IoU for tiny objects, while LEVIR shows a smaller but non-zero gap. Score-to-IoU rank correlation also weakens for the smallest buckets. This is the only hypothesis here with a coherent mechanism, a direct detector-level measurement, and replication across multiple domains.

### 6. Training-feasibility gate: post-hoc calibration fails, but candidate identity is stable
The locally best candidate is usually stable under brightness and blur perturbations, with tiny-object identity stability mostly around 0.85--0.93. Therefore the problem is not simply that candidate identity changes randomly. However, a held-out frozen calibration using score, feature energy, score-energy interaction, area, and aspect ratio does not improve top-candidate IoU: the changes are approximately -0.014 on LEVIR, -0.001 on VisDrone, and -0.016 on TinyPerson. This rejects inference-time reranking and simple score calibration as the solution. The useful remaining target is training-time responsibility assignment, where the score itself is learned from better scale-conditioned supervision.

## Recommended research direction
Prioritize **scale-conditioned local responsibility learning under stable candidate identity**. Do not build an inference reranker. Use the existing candidate set and make training assign soft responsibility according to localization utility plus augmentation consistency, with a separate tiny-object target rather than global TAL mass normalization. The method hypothesis is that candidate identity is available, but the current classification target does not teach the score to select the useful candidate. The first intervention should be a detached, uncertainty-aware responsibility target, not a new feature module.

## Falsification and next experiments
- Repeat local candidate matching over three seeds using the official validation and test protocols.
- Replace the decoded-box center neighborhood with the exact anchor/grid responsibility set used by TAL, and report the gap separately for P2/P3/P4.
- Capture true per-candidate classification, box, and DFL losses and test whether the oracle gap is caused by classification assignment, regression quality, or both.
- Run an energy-matched counterfactual that swaps only candidate classification logits while holding boxes fixed.
- Prototype only a training-time detached responsibility target, preserving the original detector and inference path.
- Compare the target against standard TAL and the already-tested marked-mass normalization. Require lower local oracle gap and no medium-object degradation on at least two datasets before any AP evaluation.

## Limitations
The matched sweep used one report-listed seed per dataset and 32 annotated images per dataset. The feasibility gate is a frozen diagnostic, not a training result. It establishes that post-hoc reranking is not promising and that the next method must change training responsibility targets. Full-seed training validation remains required.

## Decisive training-feasibility gate: TinyPerson

The first opt-in training gate was run on the report-listed TinyPerson pipeline using the same YOLOv8n P3/P4/P5 baseline, `seed=42`, fixed `split_seed=42`, `workers=8`, 640px input, and three training epochs. Only the classification responsibility target changed:

| Variant | Responsibility target | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---:|---:|---:|---:|
| B0 | standard TAL, `off` | 0.2666 | 0.0880 | 0.3123 | 0.0986 |
| B1 | per-GT normalized predicted IoU, `iou` | 0.0964 | 0.0239 | 0.1059 | 0.0257 |
| B2 | per-GT normalized sqrt-IoU, `iou_sqrt` | 0.1645 | 0.0432 | 0.1799 | 0.0482 |

The exact test protocol was the generated TinyPerson corner-window split `sw640/sh512`, with `test` pointing to the 17,693-window evaluation set. The source artifacts are the split-qualified `evaluation_metrics.json` files and `experiment_manifest.json` files in the three task-specific uploaded repositories:

- B0: `duyle2408/tinyperson-responsibility-off-smoke-v2-runs`
- B1: `duyle2408/tinyperson-responsibility-iou-smoke-v3-runs`
- B2: `duyle2408/tinyperson-responsibility-iou-sqrt-smoke-v3-runs`

Both proposed targets substantially degraded validation and test performance relative to standard TAL. B1 retained only 34% of baseline test AP50 and 26% of baseline test mAP50-95. B2 was less destructive, but still retained only 58% and 49%, respectively. This is a decisive negative result for the naive formulation:

> Replacing classification responsibility with the current predicted-IoU quality, even with a square-root softening, is not a viable first method.

This does not falsify the broader candidate-responsibility hypothesis. It falsifies the assumption that a noisy, early predicted IoU is a safe positive-classification target. The next method must use a detached teacher or augmentation-aggregated utility, must preserve a minimum objectness/classification floor, and must be evaluated first by responsibility accuracy and oracle-gap reduction before any full AP campaign. Do not run a full multi-seed sweep of B1/B2 in their current form.

## Updated decision

The recommended direction remains **scale-conditioned local responsibility learning under stable candidate identity**, but the immediate implementation target is narrower:

1. Keep standard TAL assignment and standard positive supervision as the safety path.
2. Add only a bounded residual responsibility correction, not a replacement target.
3. Construct the correction from detached, augmentation-consistent candidate utility rather than one-step predicted IoU.
4. Gate the correction by object scale and uncertainty, with an explicit fallback to B0.
5. Require no regression on medium objects, lower local oracle gap on at least two datasets, and stable val/test metrics before longer training.

The evidence now supports a focused research direction, but not a claim that the first responsibility target works.

## B3-B6 bounded residual follow-up

The requested follow-up kept the standard detector, optimizer, data split, seed, image size, batch size, workers, and NMS IoU fixed. It changed only the classification responsibility path. All runs used three remote training epochs on TinyPerson with the generated corner-window protocol `sw640/sh512`, where the test split contains 17,693 windows. The split-qualified metrics below are from each run's `evaluation_metrics.json`.

| Variant | Responsibility mode | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---:|---:|---:|---:|
| B0 | standard TAL, `off` | 0.266618 | 0.087976 | 0.312338 | 0.098611 |
| B3 | bounded detached residual | 0.217014 | 0.071044 | 0.219437 | 0.068131 |
| B4 | residual plus tiny-only gate | 0.263660 | 0.081930 | 0.309724 | 0.097945 |
| B5 | residual plus geometric consistency proxy | 0.237847 | 0.074394 | 0.284961 | 0.092150 |
| B6 | curriculum warmup then residual-consistent mode | 0.266618 | 0.087976 | 0.312338 | 0.098611 |

The uploaded source artifacts are task-specific repositories:

- B3: `duyle2408/tinyperson-responsibility-b3-smoke-runs`
- B4: `duyle2408/tinyperson-responsibility-b4-smoke-runs`
- B5: `duyle2408/tinyperson-responsibility-b5-smoke-runs`
- B6: `duyle2408/tinyperson-responsibility-b6-smoke-runs`

Each repository upload marker lists `evaluation_metrics.json`, `experiment_manifest.json`, `results.csv`, and both checkpoints as remotely verified. The merged TinyBenchmark evaluator was unavailable because `pycocotools` was not installed. The regular split-qualified validation and corner-window test metrics are available and must not be confused with merged-test metrics.

### Follow-up decision

The smoke matrix gives a clear safety result, but not evidence of an AP improvement:

- B3 is unsafe as configured. A bounded residual from the beginning still damages test AP50 by 29.7% relative to B0.
- B4 is the safest non-control variant. It is within 0.0026 test AP50 and 0.0007 test mAP50-95 of B0, indicating that a tiny-only gate prevents broad-object damage. This is a stability signal, not a gain.
- B5 is better than B3 but remains below B0. Its consistency term is a deterministic geometric perturbation proxy, not true paired-view augmentation. It should not be presented as validated augmentation consistency.
- B6 exactly matches B0 in this three-epoch smoke. This means the warmup and ramp avoided the early-training collapse, but it also means the correction had no measurable effect at this schedule. B6 is a safe curriculum candidate, not a demonstrated improvement.

The decision is therefore narrowed again: continue only with **B4/B6-style gated residual responsibility as a diagnostic**, and do not start a full method sweep or claim a new detector method yet. The next decisive experiment should use a longer matched schedule and log local oracle gap, responsibility entropy, positive mass per GT, and medium-object metrics by epoch. If those diagnostics do not improve while AP remains at control level, discard this direction. If the correction reduces the oracle gap without harming B0-level metrics on VisDrone and TinyPerson, then proceed to true paired augmentation views and multi-seed validation. LEVIR remains an important negative/control domain because its original oracle gap is small.

## Long TinyPerson validation and architecture transfer

The three-epoch smoke matrix above was followed by matched 100-epoch TinyPerson runs using the native corner-window protocol `sw640/sh512`, fixed `seed=42`, fixed `split_seed=42`, `batch=8`, `workers=8`, `imgsz=640`, and `NMS IoU=0.5`. The test split is the dataset YAML `test` path for the generated corner-window dataset. These values are not merged TinyBenchmark metrics.

### Long YOLOv8 runs

| Variant | Mode | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---:|---:|---:|---:|
| B0 | standard TAL | 0.443810 | 0.155491 | 0.460551 | 0.161225 |
| B4 | tiny-only bounded residual | 0.505491 | 0.180482 | 0.501943 | 0.182564 |
| B6 | warmup/ramp into residual curriculum | 0.508399 | 0.180783 | 0.500664 | 0.179607 |

The long schedule changes the interpretation of the smoke result. B4 improves over the matched YOLOv8 B0 on both validation and test AP50, while B6 gives a similar validation result and slightly lower test mAP50-95 than B4. This supports keeping the gated residual mechanism as a serious diagnostic candidate on TinyPerson, but it is still one dataset and one seed. The improvement is not evidence that the mechanism is architecture-independent.

### YOLOv9t and YOLOv10n transfer runs

To test transfer beyond YOLOv8, B4 was run with `yolov9t.pt` and `yolov10n.pt` under the same TinyPerson data protocol and responsibility hyperparameters. The architecture-transfer runs used their own task-specific Hugging Face repositories and their upload markers were publicly verified. The corresponding YOLOv9/YOLOv10 B0 controls were not run, so comparisons against YOLOv8 B0 are exploratory and confounded by architecture.

| Architecture | Variant | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv9t | B4 residual tiny | 0.510144 | 0.179763 | 0.501187 | 0.178271 |
| YOLOv10n | B4 residual tiny | 0.459874 | 0.168860 | 0.434883 | 0.159999 |

The YOLOv9t result is numerically close to the YOLOv8 B4 result, but without a YOLOv9t B0 it cannot establish a treatment effect. YOLOv10n B4 is weaker on test AP50 than both YOLOv8 B0 and YOLOv8 B4. Therefore the current evidence does **not** support calling B4 architecture-agnostic. It supports a narrower claim: the gated residual responsibility path is compatible with multiple detector implementations, while its benefit must be measured against an architecture-matched TAL control.

Source artifacts:

- YOLOv8 long B4/B6: `duyle2408/tod-responsibility-b4-b6-rerun-runs`, prefixes `runs/tinyperson/residual_tiny/seed_42` and `runs/tinyperson/residual_curriculum/seed_42`.
- YOLOv9t B4: `duyle2408/tod-responsibility-yolov9-b4-rerun-runs`, prefix `runs/tinyperson/residual_tiny/seed_42`.
- YOLOv10n B4: `duyle2408/tod-responsibility-yolov10-b4-runs`, prefix `runs/tinyperson/residual_tiny/seed_42`.

### Updated transfer decision

Keep the research direction, but do not broaden the claim yet. The strongest current result is the matched long YOLOv8 TinyPerson comparison, where B4/B6 improve validation and B4 improves test AP50. The YOLOv9 transfer is promising but lacks an architecture-matched B0. The YOLOv10 transfer is negative relative to the available YOLOv8 controls. The next required experiment is therefore not another variant: run **YOLOv9t B0 and YOLOv10n B0**, then compare each architecture's B4 against its own TAL baseline using local oracle-gap and responsibility-alignment diagnostics. If B4 fails to reduce the mechanism metrics against both matched controls, stop the transfer claim and retain B4 only as a YOLOv8/TinyPerson-specific lead.
