# SQX Custom Analysis extensions

| Snippet | What it does |
|---|---|
| [`WFMStability`](#wfmstability) | Filters out strategies whose Walk-Forward Matrix 3D chart is unstable, judged by scale rather than by an absolute difference. |
| [`RemoveExitAfterBars`](#removeexitafterbars) | Disables the Exit After Bars block on every strategy of a databank, replacing the old external-script round trip. |
| [`SelectBestByIndicatorGroup`](#selectbestbyindicatorgroup) | Keeps the best strategy of each unique **Entry + Price** indicator combination. |
| [`RenameStrategies`](#renamestrategies) | Renames `Strategy 12345` to `EURUSD H1 Long 12345` across the whole databank. |

## Install

Copy the snippets into your SQX user folder (paths mirror SQX's own layout):

```
src/SQX_extend/Snippets/SQ/CustomAnalysis/*.java
        ->  <SQX_PATH>/user/extend/Snippets/SQ/CustomAnalysis/
```

With the current `config.ini` (`SQX_PATH=C:\SQX_144`) that is:

```
C:\SQX_144\user\extend\Snippets\SQ\CustomAnalysis\
```

Restart SQX (or use *Tools → Recompile snippets*). Each then appears under its
own name in the Custom Analysis list of any databank.

---

# WFMStability

Scale-free stability filter for the Walk-Forward Matrix 3D chart. Keeps
strategies whose WFM surface is **flat**, and rejects the unstable ones —
without you having to know the right absolute number up front.

## Why the colleague's version needs retuning and this one does not

`WFMStagnationStability50` and friends decide flatness with a **difference**:

```java
zoneMax - zoneMin <= 50.0     // 50 what? days of stagnation
```

That number is tied to the scale of the strategy being measured. Two equally
flat surfaces get opposite verdicts purely because of where they sit:

| Surface        | max − min | max / min | `max-min <= 50` | WFMStability |
|----------------|-----------|-----------|-----------------|--------------|
| 200 … 250      | 50        | 1.25      | passes          | **rejected** |
| 2000 … 2050    | 50        | 1.025     | passes          | **passes**   |
| 200 … 300      | 100       | 1.50      | rejected        | **rejected** |
| 2000 … 2100    | 100       | 1.05      | rejected        | **passes**   |

The last row is the expensive one: a surface varying by 5% is flat by any
reasonable reading, but an absolute filter throws it away. That is why the
thresholds had to be forked into `...Stability5 / 50 / 75 / 100` — one per
scale — and why you still have to guess which to use per databank.

This is not a new idea, it is what hobbiecode actually says:

> "Hay que tener en cuenta la escala […] si los valores de la escala son muy
> cercanos, si quitamos zoom al gráfico va a ser prácticamente plano. Cuando los
> valores de la escala ya pasan a ser **el doble o el triple** y vemos que no hay
> zonas planas, es cuando tenemos que descartar la ventaja."

"El doble o el triple" is a **ratio**, not a difference. Measuring `max / min`
is what makes one setting work on every symbol, timeframe and metric.

## What it checks

Both checks are derived from the strategy's own WFM surface:

1. **Global scale** — `globalMax / globalMin <= maxGlobalRatio` (default `2.0`).
   Directly encodes *"que la escala del máximo al mínimo no sea del doble"*.
2. **Stable zone** — some `zone x zone` window (default **3x3**) where
   `zoneMax / zoneMin <= maxZoneRatio` (default `1.20`, i.e. 20% spread).
   Encodes *"que haya un área 3x3 estable"*.

Plus a **quality** guard so a flat-but-worthless corner cannot sneak through:
the zone mean must stay within `qualityMult` of the surface's best value
(default `2.0`). For Stagnation "best" is the minimum, for Ret/DD the maximum —
the direction flips automatically per metric, so one number means the same thing
either way.

A strategy passes only if **all** requested metrics pass both checks.

## Arguments

Typed into the Custom Analysis parameter box, comma separated. All optional —
omit any trailing ones.

```
metric[+metric], zone, maxZoneRatio, maxGlobalRatio, qualityMult [, debug]
```

| Argument         | Default      | Meaning |
|------------------|--------------|---------|
| `metric`         | `Stagnation` | Any SQX stat name. Join with `+` to require all. |
| `zone`           | `3`          | Size of the stable square (hobbiecode's 3x3). |
| `maxZoneRatio`   | `1.20`       | Max `max/min` inside the zone. Lower = stricter. |
| `maxGlobalRatio` | `2.00`       | Max `max/min` over the whole surface. `99` disables. |
| `qualityMult`    | `2.00`       | Zone mean must stay within this factor of the global best. `0` disables. |
| `debug`          | off          | Add the word `debug` anywhere to dump the full matrix to the project log. |

### Examples

```
                                   Stagnation, 3x3, 20% zone spread, 2x global
ReturnDDRatio                      same thresholds, on Ret/DD
Stagnation+ReturnDDRatio           both must have a stable 3x3 zone
Stagnation, 3, 1.10, 2.0, 2.0      stricter: only 10% spread inside the zone
Stagnation, 3, 1.20, 3.0, 2.0      hobbiecode's looser "triple" global gate
Stagnation, 4, 1.25, 99, 2.0       4x4 zone, no global gate (zone check only)
Stagnation, 3, 1.20, 2.0, 2.0, debug   same, with the matrix in the log
```

## Reading the log

Every strategy writes one line to the project log explaining the verdict, so you
can see *why* something was dropped instead of guessing:

```
PASS   EURUSD H1 Long: Stagnation OK (stable)
REJECT XAUUSD H4 Short: Stagnation FAIL (surface spread 5.00x (max 2.00x) and no flat 3x3 zone)
REJECT GBPUSD H1 Long: ReturnDDRatio FAIL (flattest qualifying zone is 1.41x (max 1.20x))
```

With `debug` you also get the surface, with the flattest qualifying zone marked:

```
  Stagnation surface (runs x OOS%), 36 values, min 750.00 max 1082.00, spread 1.44x
              26%      28%      30%      32%      34%      36%
     5r   1082.00  1075.00  1070.00  1065.00  1060.00  1055.00
     6r   1080.00 [1074.00][1069.00][1064.00]  1059.00  1054.00
     7r   1078.00 [1073.00][1068.00][1063.00]  1058.00  1053.00
     8r    860.00 [ 855.00][1067.00][1062.00]  1057.00  1052.00
```

## Tuning

Start with the defaults and read the log. Then:

- **Too many strategies survive** → lower `maxZoneRatio` (`1.20` → `1.10`).
- **Everything is rejected** → raise `maxGlobalRatio` to `3.0` (hobbiecode
  accepts "doble o triple"), or disable it with `99` and rely on the 3x3 zone.
- **Flat but poor strategies survive** → lower `qualityMult` (`2.0` → `1.5`).

`maxZoneRatio` is still a number you choose, but unlike `MAX_RANGE = 50` it is
**scale-free**: `1.20` means "20% variation" on every databank, every symbol and
both metrics. It does not need to be re-derived when the scale changes.

The global gate *is* auto-derived from the max and min of the whole WFM, exactly
as requested. The zone threshold is deliberately not: deriving it from the
surface's own spread would make a wilder surface get a *more* lenient bar, which
is backwards.

## Notes

- Reads `Directions.Both / PlTypes.Money / SampleTypes.FullSample`, falling back
  to `OutOfSample` — the same source the 3D chart plots.
- Runs missing a value leave a gap; any window containing one is skipped.
- Ret/DD surfaces that touch or cross zero are rejected: a ratio is meaningless
  there, and a losing WF run is not a stable one.
- Requires a WFM to have been run — strategies with no `WF:` results are dropped.
- Verified: compiles against `SQTradingLib.jar` (SQX 144), and the scale-free
  math is covered by 43 assertions over synthetic WFM surfaces.

---

# RemoveExitAfterBars

Disables the Exit After Bars block on every strategy of a databank, in place.

This replaces the old `sqx_tool.py remove_eab` step. That one had to run as a
`CallExternalScript` task: SQX saved `.sqx` files to a folder, Python rewrote
them, and `LoadFromFiles` read them back. All of that disappears — the edit now
happens inside SQX, in memory.

## Usage

Add it as a **Custom Analysis** on the databank, before the Retest task that
should measure the strategies without the bar exit.

| Input args | Effect |
|------------|--------|
| *(empty)*  | Disable Exit After Bars |
| `EAB=0`    | Disable Exit After Bars |
| `NoEAB`    | Disable Exit After Bars |
| `EAB=12`   | Force Exit After Bars to 12 bars on every strategy |
| `12`       | Same, bare number |

Tokens that belong to other analyses (`SL=`, `TP=`, …) are ignored, so the same
argument string can be shared with `CAAddFixedSLTP` unchanged.

## How it disables the block

Exit After Bars is a Param on each entry block, and the block declares
`minValue="0"` / `defaultValue="0"` — so **0 is its own "do nothing" value**:

```xml
<Param key="#ExitAfterBars.ExitAfterBars#" type="int" minValue="0"
       defaultValue="0" ...>12</Param>
```

Setting it to 0 disables the exit while leaving the XML structurally intact.
That is deliberately safer than deleting the Param, as the Python version did:
nothing can end up referencing a node that no longer exists.

The Param comes in two shapes and both are handled:

- **literal value** — the text is replaced with `0`
- **variable reference** — carries `variable="true"` with a `<variable>` id as
  its text. The attribute is dropped so it becomes a plain literal `0`, and the
  shared `<variable>` is left alone for whatever else may reference it.

Running it twice is a no-op the second time.

> **Order matters:** editing the strategy XML does not re-run the backtest. The
> databank metrics keep describing the strategies as they were until a Retest
> task runs *after* this one.

## Migrating from the Python step

`remove_eab` / `remove_eab_b64` are gone from `sqx_tool.py`, along with the
`CallExternalScript-Task1` and `LoadFromFiles-Task1` wiring in `newproject`.
`CallExternalScript-Task2` (the file rename) is untouched and still works.

The template in `src/Template/` still ships `CallExternalScript-Task1.xml` and
`LoadFromFiles-Task1.xml`; `newproject` simply no longer patches them. Remove
them from the template and add this Custom Analysis in their place.

---

# SelectBestByIndicatorGroup

Deduplicates a databank by indicator combination. Two strategies count as the
same edge **only when their Entry indicators *and* their Price indicators
match**; of each such group the highest fitness survives.

## Why not group on Entry indicators alone

`SelectBestByEntryGroupArgs` keys on the Entry indicators only. Those two SQX
columns are complementary:

- **Entry indicators** — blocks used in the entry *conditions*
- **Price indicators** — blocks used in the entry *price levels* (Enter at Stop / Limit)

So a databank containing these two:

| Strategy | Entry indicators | Price indicators | Fitness |
|----------|------------------|------------------|---------|
| A        | `MA,RSI`         | `BB`             | 1.0     |
| B        | `MA,RSI`         | `MA`             | 0.9     |

is two different edges — one enters at a Bollinger band, the other at a moving
average. Grouping on Entry alone sees one group and **throws B away**. Grouping
on both keeps them apart.

Both keys come from SQX's own `EntryIndicators` / `PriceIndicators` databank
columns, so grouping matches exactly what those columns show in the UI. Each
builds its string through a `TreeSet`, so it is already sorted and
de-duplicated — `MA,RSI` regardless of the order the blocks appear in the
strategy — and equal indicator sets always produce an equal key.

## Usage

Add as a **Full databank analysis** task in a Custom Project:

- Set the **Source** databank (strategies to evaluate)
- Set the **Target** databank (winners are copied here)
- Turn **off** *"Filter by results of custom analysis"*

| Input args | Effect |
|------------|--------|
| *(empty)*  | Keep the best 1 per group by **Fitness** |
| `Fitness,2`| Keep the best 2 per group by Fitness |
| `RetDD`    | Keep the best 1 per group by Ret/DD, full sample |
| `RetDD,3,OOS` | Keep the best 3 per group by Ret/DD, out of sample |
| `Drawdown` | Keep the *lowest* drawdown per group |

Format is `Criterion,N,Sample` — same shape as `SelectBestByEntryGroupArgs`.

- **Criterion**: `Fitness` (default), `RetDD`, `Sharpe`, `ProfitFactor`,
  `NetProfit`, `Calmar`, `WinRate`, `Drawdown` (smaller is better)
- **N**: how many to keep per group, `1`–`10` (default `1`)
- **Sample**: `Full` (default), `IS`, `OOS`. Ignored for `Fitness`, which is a
  single value per strategy.

## Behaviour worth knowing

- **Unidentifiable strategies always survive.** If either column returns `N/A`
  (e.g. the strategy XML cannot be read), the strategy gets a one-off key and is
  never treated as anyone's duplicate. Silently dropping a strategy we failed to
  identify is the one unrecoverable mistake this filter could make, so it errs
  toward keeping.
- **Empty is not the same as unavailable.** A strategy that genuinely has no
  price indicators (enters at market) has an empty Price value, and those group
  together normally. Only `N/A` means "unknown".
- **Ties keep the databank's order**, and output preserves the original
  ordering, so repeated runs give identical results.
- **NaN scores rank last** — a strategy whose criterion could not be computed
  never displaces one that has a real score.

## Notes

- Verified: compiles against `SQTradingLib.jar` / `Snippets.jar` (SQX 144), with
  42 assertions over the grouping, top-N, tie-breaking and argument handling.

---

# RenameStrategies

Gives strategies a readable name inside the databank. Two independent edits,
applied in this order:

1. the leading `Strategy` word is replaced with the input text
2. the word `Improved` is dropped from the separator SQX inserts

```
E-Build   Strategy 12345                        ->  XAUUSD H1 Long 12345
S-Build   XAUUSD H1 Long 12345 - Improved 678   ->  XAUUSD H1 Long 12345 - 678
```

**Step 2 runs whether or not any input text was given**, which is what lets the
same snippet serve both stages: pass the prefix at E-Build, leave it empty at
S-Build where the name is already right and only `Improved` needs to go.

## Usage

Works as **either** a per-strategy or a full-databank analysis — pick whichever
slot suits the task. Run it before the `SaveToFiles` task whose output you want
named.

| Stage | Input args | Effect |
|-------|------------|--------|
| E-Build | `XAUUSD H1 Long` | Replace `Strategy` with this text, and drop `Improved` |
| S-Build | *(empty)* | Only drop `Improved`, leave the rest of the name alone |

Passing the prefix at S-Build too is harmless — by then no name starts with
`Strategy `, so step 1 finds nothing to do and only `Improved` is removed.

### Per strategy or full databank

Registered as `TYPE_BOTH`, so it appears in both dropdowns of the Custom
Analysis task. SQX's own UI code decides that:

```js
// SettingsCustomAnalysisService.js
if(method.type==20 || method.type==30) ... availableMethodsPerStrategy.push(method);
if(method.type==10 || method.type==30) ... availableMethodsFullDatabank.push(method);
```

Both give the identical result. SQX hands the *same* `ResultsGroup` objects to
either hook and keeps using those objects afterwards — the task threads one list
through `perStrategy1` → `fullDatabank1` → `perStrategy2` → `fullDatabank2` and
then adds them to the output databank — so a rename applied in either place
sticks.

One caveat that shaped the code: in per-strategy mode SQX **removes** any
strategy whose `filterStrategy` returns `false`. This one therefore always
returns `true`, even when a rename fails. It renames; it never filters.

## Behaviour worth knowing

- **Running twice is a no-op.** Once a name no longer starts with `Strategy `
  and holds no `Improved`, both steps do nothing and no write happens. Safe to
  re-run, or to leave in a project that loops.
- **Every `Improved` is dropped**, not just the first, so a twice-improved
  `... 1 - Improved 2 - Improved 3` ends up as `... 1 - 2 - 3`. This differs
  from the old `rename_files`, whose non-greedy regex only ever collapsed one
  and left `... 1 - 2 - Improved 3`. The plain reading of "remove Improved" won.
- **Names that are not SQX defaults are left alone.** `My Portfolio` stays
  `My Portfolio`; only the exact `Strategy ` prefix is replaced.
- The trailing id is always preserved, so strategies stay distinguishable.

## Replaces rename_files

`rename_files`, `rename_files_b64` and the `CallExternalScript-Task2` wiring
have been removed from `sqx_tool.py`. Renaming the strategy itself puts the
name everywhere it shows up — the databank, the exported `.sqx` / `.mq5` files
(SQX names them after the strategy), and the generated source — instead of only
in the folders that script happened to visit.

The template still ships `CallExternalScript-Task2.xml`; `newproject` simply no
longer patches it.

## Notes

- Verified: compiles against `SQTradingLib.jar` (SQX 144). 25 assertions cover
  the E-Build / S-Build pipeline, repeated improvements, idempotency and the
  leave-alone cases; 11 more drive both entry points against a real
  `ResultsGroup` and confirm they produce identical names.
