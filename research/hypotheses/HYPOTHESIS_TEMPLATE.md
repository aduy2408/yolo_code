# H000: <short mechanism name>

## Claim

<What should improve, for which object/dataset regime?>

## Mechanism

<What training, architecture, data, or inference mechanism causes the improvement?>

## Evidence already available

- <report, probe, or prior run>

## Observable prediction

- Primary diagnostic: <metric> should <direction>.
- Secondary diagnostic: <metric> should <direction>.
- Final metric: <metric> should <direction>.

## Falsification

Reject if:

- <unsafe metric or boundary>
- <diagnostic contradicts the mechanism>
- <protocol or implementation condition>

## Minimal experiment

- Baseline:
- Treatment:
- Single changed factor:
- Dataset:
- Training seed:
- Split seed: 42
- Image size:
- Smoke schedule:
- Full schedule, if promoted:
- NMS IoU: 0.5

## Budget

- Maximum treatment variants: 3
- Maximum seeds before promotion: 2
- Maximum datasets before promotion: 2

## Decision history

| Experiment | Result | Decision | Reason |
|---|---|---|---|
| — | — | — | — |
