package SQ.CustomAnalysis;

import com.strategyquant.lib.*;
import com.strategyquant.datalib.*;
import com.strategyquant.tradinglib.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * WFMStability - scale-free stability filter for the Walk-Forward Matrix 3D chart.
 *
 * Keeps only strategies whose WFM surface is FLAT, judged the way hobbiecode
 * describes it: by SCALE, not by absolute difference.
 *
 *   "Hay que tener en cuenta la escala [...] si los valores de la escala son muy
 *    cercanos, si quitamos zoom al grafico va a ser practicamente plano. Cuando
 *    los valores de la escala ya pasan a ser el doble o el triple y vemos que no
 *    hay zonas planas, es cuando tenemos que descartar la ventaja."
 *
 * "El doble o el triple" is a RATIO (max / min), not a difference. That is what
 * makes this filter work on any databank without retuning: a strategy whose
 * Stagnation runs 200..240 and one whose Stagnation runs 1000..1200 are equally
 * flat (both 1.2x) and both pass, while a fixed "max - min <= 50" would accept
 * the first and reject the second purely because of its scale.
 *
 * Two independent checks, both derived from the strategy's own WFM surface:
 *
 *   1. GLOBAL SCALE  - globalMax / globalMin <= maxGlobalRatio.
 *                      "que la escala del maximo al minimo no sea del doble".
 *   2. STABLE ZONE   - some zone x zone window (3x3 by default) where
 *                      zoneMax / zoneMin <= maxZoneRatio, and whose mean is
 *                      still close to the best value on the whole surface.
 *                      "que haya un area 3x3 estable".
 *
 * The quality check stops a flat-but-worthless corner from passing: the zone
 * mean must stay within qualityMult of the surface's best value, in whichever
 * direction is "better" for that metric.
 *
 * INPUT ARGUMENTS (comma separated, all optional, trailing ones can be omitted):
 *
 *   metric[+metric], zone, maxZoneRatio, maxGlobalRatio, qualityMult [, debug]
 *
 *   metric          SQX stat name(s). Join with '+' to require ALL of them.
 *                   Default: Stagnation
 *   zone            Size of the stable square. Default: 3  (hobbiecode's 3x3)
 *   maxZoneRatio    Max max/min inside the zone. Default: 1.20 (20% spread)
 *   maxGlobalRatio  Max max/min over the whole surface. Default: 2.00
 *                   Set very high (e.g. 99) to rely on the zone check alone.
 *   qualityMult     Zone mean must stay within this factor of the global best.
 *                   Default: 2.00. Set 0 to disable.
 *   debug           Add the word "debug" anywhere to dump the full matrix.
 *
 * EXAMPLES:
 *
 *   (empty)                                 Stagnation, 3x3, 1.20, 2.00, 2.00
 *   ReturnDDRatio                           same thresholds, on Ret/DD
 *   Stagnation+ReturnDDRatio                both must have a stable 3x3 zone
 *   Stagnation, 3, 1.10, 2.0, 2.0           stricter zone (10% spread)
 *   Stagnation, 4, 1.25, 99, 2.0, debug     4x4 zone, no global gate, verbose
 *
 * Deploy to: <SQX>/user/extend/Snippets/SQ/CustomAnalysis/WFMStability.java
 */
public class WFMStability extends CustomAnalysisMethod {

    private static final String DEF_METRICS       = "Stagnation";
    private static final int    DEF_ZONE          = 3;
    private static final double DEF_ZONE_RATIO    = 1.20;
    private static final double DEF_GLOBAL_RATIO  = 2.00;
    private static final double DEF_QUALITY_MULT  = 2.00;

    /** Metrics where a SMALLER value is better. Anything else: bigger is better. */
    private static final String[] LOWER_IS_BETTER = {
        "stagnation", "maxdd", "maxddpercent", "maxddpct", "drawdown",
        "averagedd", "avgdd", "maxdrawdown"
    };

    /** WFM result keys look like "WF: 6 runs : 28 % OOS". */
    private static final Pattern RUNS_RE = Pattern.compile("(\\d+)\\s*runs");
    private static final Pattern OOS_RE  = Pattern.compile("(\\d+)\\s*%");

    public WFMStability() {
        super("WFMStability", TYPE_FILTER_STRATEGY);
    }

    // ------------------------------------------------------------------
    //  Configuration
    // ------------------------------------------------------------------

    static class Config {
        String[] metrics       = { DEF_METRICS };
        int      zone          = DEF_ZONE;
        double   maxZoneRatio  = DEF_ZONE_RATIO;
        double   maxGlobalRatio = DEF_GLOBAL_RATIO;
        double   qualityMult   = DEF_QUALITY_MULT;
        boolean  debug         = false;
    }

    static Config parseArgs(String raw) {
        Config c = new Config();
        if (raw == null) return c;

        // "debug" may appear anywhere; strip it before positional parsing.
        String s = raw.trim();
        if (s.toLowerCase(Locale.US).contains("debug")) {
            c.debug = true;
            s = s.replaceAll("(?i),?\\s*debug\\s*", "");
        }
        s = s.trim();
        if (s.isEmpty()) return c;

        String[] parts = s.split(",");
        if (parts.length > 0 && !parts[0].trim().isEmpty()) {
            String[] metrics = parts[0].trim().split("\\+");
            ArrayList<String> cleaned = new ArrayList<String>();
            for (int i = 0; i < metrics.length; i++) {
                String m = metrics[i].trim();
                if (!m.isEmpty()) cleaned.add(m);
            }
            if (!cleaned.isEmpty()) c.metrics = cleaned.toArray(new String[0]);
        }
        if (parts.length > 1) c.zone           = (int) parseOr(parts[1], c.zone);
        if (parts.length > 2) c.maxZoneRatio   = parseOr(parts[2], c.maxZoneRatio);
        if (parts.length > 3) c.maxGlobalRatio = parseOr(parts[3], c.maxGlobalRatio);
        if (parts.length > 4) c.qualityMult    = parseOr(parts[4], c.qualityMult);

        if (c.zone < 2) c.zone = 2;
        return c;
    }

    private static double parseOr(String raw, double fallback) {
        try {
            String s = raw.trim().replace(',', '.');
            if (s.isEmpty()) return fallback;
            return Double.parseDouble(s);
        } catch (Exception e) {
            return fallback;
        }
    }

    static boolean isHigherBetter(String metric) {
        String m = metric.toLowerCase(Locale.US);
        for (int i = 0; i < LOWER_IS_BETTER.length; i++) {
            if (LOWER_IS_BETTER[i].equals(m)) return false;
        }
        return true;
    }

    // ------------------------------------------------------------------
    //  Scale-free math  (static + self-contained, so it can be unit tested)
    // ------------------------------------------------------------------

    /**
     * Scale-free spread of a range: max / min.
     *
     * A ratio only means something for strictly positive values, so:
     *   - a range that collapses to a single value is flat            -> 1.0
     *   - a range touching or crossing zero has no meaningful ratio   -> +Inf
     * Returning +Inf makes those surfaces fail every threshold, which is the
     * safe direction for a filter.
     */
    static double spreadRatio(double min, double max) {
        if (Double.isNaN(min) || Double.isNaN(max)) return Double.NaN;
        if (min == max)  return 1.0;
        if (min <= 0.0)  return Double.POSITIVE_INFINITY;
        return max / min;
    }

    /** min, max and mean of a zone x zone window, or null if it contains a gap. */
    static double[] zoneStats(double[][] m, int row, int col, int zone) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double sum = 0.0;
        for (int r = row; r < row + zone; r++) {
            for (int c = col; c < col + zone; c++) {
                double v = m[r][c];
                if (Double.isNaN(v)) return null;
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
            }
        }
        return new double[] { min, max, sum / (zone * zone) };
    }

    /** Is the zone mean still close enough to the best value on the surface? */
    static boolean qualityOk(double zoneMean, double globalBest,
                             boolean higherIsBetter, double mult) {
        if (mult <= 0.0) return true;                        // check disabled
        if (Double.isNaN(zoneMean) || Double.isNaN(globalBest)) return false;
        return higherIsBetter ? zoneMean >= globalBest / mult
                              : zoneMean <= globalBest * mult;
    }

    static class Verdict {
        boolean pass;
        String  reason        = "";
        double  globalMin     = Double.NaN;
        double  globalMax     = Double.NaN;
        double  globalRatio   = Double.NaN;
        boolean globalOk;
        boolean zoneFound;
        double  bestZoneRatio = Double.NaN;
        double  bestZoneMean  = Double.NaN;
        int     bestRow       = -1;
        int     bestCol       = -1;
        int     validCells;
    }

    /**
     * Decide whether one metric's surface is stable.
     *
     * Everything is derived from the surface itself - no absolute thresholds -
     * so the same settings work across symbols, timeframes and metrics.
     */
    static Verdict evaluate(double[][] m, boolean higherIsBetter, Config c) {
        Verdict v = new Verdict();

        int rows = m.length;
        int cols = (rows == 0) ? 0 : m[0].length;

        double gMin = Double.POSITIVE_INFINITY;
        double gMax = Double.NEGATIVE_INFINITY;
        for (int r = 0; r < rows; r++) {
            for (int col = 0; col < cols; col++) {
                double val = m[r][col];
                if (Double.isNaN(val)) continue;
                if (val < gMin) gMin = val;
                if (val > gMax) gMax = val;
                v.validCells++;
            }
        }

        if (v.validCells == 0) {
            v.reason = "no WF runs reported this metric";
            return v;
        }

        v.globalMin = gMin;
        v.globalMax = gMax;

        // A "bigger is better" metric whose best run is still <= 0 is broken,
        // and its ratios would be meaningless. Reject before doing the math.
        if (higherIsBetter && gMax <= 0.0) {
            v.reason = "best WF run is not positive (" + fmt(gMax) + ")";
            return v;
        }

        v.globalRatio = spreadRatio(gMin, gMax);
        v.globalOk    = v.globalRatio <= c.maxGlobalRatio;

        if (rows < c.zone || cols < c.zone) {
            v.reason = "WFM grid " + rows + "x" + cols
                     + " is smaller than the " + c.zone + "x" + c.zone + " zone";
            return v;
        }

        double globalBest = higherIsBetter ? gMax : gMin;

        // Flattest window that still passes the quality check. If that one is
        // not flat enough, no qualifying window is.
        for (int r = 0; r <= rows - c.zone; r++) {
            for (int col = 0; col <= cols - c.zone; col++) {
                double[] st = zoneStats(m, r, col, c.zone);
                if (st == null) continue;                    // window has a gap

                if (!qualityOk(st[2], globalBest, higherIsBetter, c.qualityMult)) continue;

                double ratio = spreadRatio(st[0], st[1]);
                if (Double.isNaN(ratio)) continue;

                if (Double.isNaN(v.bestZoneRatio) || ratio < v.bestZoneRatio) {
                    v.bestZoneRatio = ratio;
                    v.bestZoneMean  = st[2];
                    v.bestRow       = r;
                    v.bestCol       = col;
                }
            }
        }

        v.zoneFound = !Double.isNaN(v.bestZoneRatio) && v.bestZoneRatio <= c.maxZoneRatio;
        v.pass      = v.globalOk && v.zoneFound;

        if (v.pass) {
            v.reason = "stable";
        } else if (!v.globalOk && !v.zoneFound) {
            v.reason = "surface spread " + fmtRatio(v.globalRatio) + "x (max "
                     + fmtRatio(c.maxGlobalRatio) + "x) and no flat "
                     + c.zone + "x" + c.zone + " zone";
        } else if (!v.globalOk) {
            v.reason = "surface spread " + fmtRatio(v.globalRatio) + "x exceeds "
                     + fmtRatio(c.maxGlobalRatio) + "x";
        } else if (Double.isNaN(v.bestZoneRatio)) {
            v.reason = "no complete " + c.zone + "x" + c.zone
                     + " zone passes the quality check";
        } else {
            v.reason = "flattest qualifying zone is " + fmtRatio(v.bestZoneRatio)
                     + "x (max " + fmtRatio(c.maxZoneRatio) + "x)";
        }
        return v;
    }

    // ------------------------------------------------------------------
    //  SQX entry point
    // ------------------------------------------------------------------

    @Override
    public boolean filterStrategy(String project, String task, String databankName,
                                  ResultsGroup rg) throws Exception {

        Config c = parseArgs(getInputArgs());
        StringBuilder log = new StringBuilder();
        log.append(rg.getName()).append(": ");

        List<String> allKeys = rg.getResultKeys();
        if (allKeys == null || allKeys.isEmpty()) {
            setProjectLog(log.append("rejected - no result keys").toString());
            return false;
        }

        // Axis values present in this WFM, collected once and shared by metrics.
        ArrayList<String>  wfKeys   = new ArrayList<String>();
        ArrayList<int[]>   coords   = new ArrayList<int[]>();
        ArrayList<Integer> runsVals = new ArrayList<Integer>();
        ArrayList<Integer> oosVals  = new ArrayList<Integer>();

        for (int i = 0; i < allKeys.size(); i++) {
            String key = allKeys.get(i);
            if (key == null || !key.startsWith("WF:")) continue;
            int[] rc = parseRunsAndOos(key);
            if (rc == null) continue;
            wfKeys.add(key);
            coords.add(rc);
            if (!runsVals.contains(rc[0])) runsVals.add(rc[0]);
            if (!oosVals.contains(rc[1]))  oosVals.add(rc[1]);
        }

        if (wfKeys.isEmpty()) {
            setProjectLog(log.append("rejected - no WF results (run a Walk-Forward Matrix first)").toString());
            return false;
        }

        Collections.sort(runsVals);
        Collections.sort(oosVals);

        boolean pass = true;
        for (int mi = 0; mi < c.metrics.length; mi++) {
            String metric = c.metrics[mi];

            double[][] matrix = buildMatrix(rg, wfKeys, coords, runsVals, oosVals, metric);
            Verdict v = evaluate(matrix, isHigherBetter(metric), c);

            if (mi > 0) log.append(" | ");
            log.append(metric).append(' ')
               .append(v.pass ? "OK" : "FAIL").append(" (").append(v.reason).append(')');

            if (c.debug) {
                log.append('\n').append(dump(matrix, runsVals, oosVals, metric, v, c));
            }

            if (!v.pass) pass = false;
        }

        setProjectLog(log.insert(0, pass ? "PASS  " : "REJECT ").toString());
        return pass;
    }

    /** Reads one metric off every WF run into a runs x oos matrix. */
    private double[][] buildMatrix(ResultsGroup rg, ArrayList<String> wfKeys,
                                   ArrayList<int[]> coords, ArrayList<Integer> runsVals,
                                   ArrayList<Integer> oosVals, String metric) {

        double[][] matrix = new double[runsVals.size()][oosVals.size()];
        for (int r = 0; r < matrix.length; r++) {
            for (int c = 0; c < matrix[r].length; c++) matrix[r][c] = Double.NaN;
        }

        for (int i = 0; i < wfKeys.size(); i++) {
            int row = runsVals.indexOf(coords.get(i)[0]);
            int col = oosVals.indexOf(coords.get(i)[1]);
            if (row < 0 || col < 0) continue;

            Result result = null;
            try {
                result = rg.subResult(wfKeys.get(i));
            } catch (Exception e) {
                continue;
            }
            if (result == null) continue;

            matrix[row][col] = readStat(result, metric);
        }
        return matrix;
    }

    /**
     * Same source the 3D chart plots: full sample, both directions, money.
     * Falls back to the out-of-sample stats when a run has no full-sample set.
     */
    private static double readStat(Result result, String metric) {
        SQStats stats = result.statsOrNull(Directions.Both, PlTypes.Money, SampleTypes.FullSample);
        if (stats == null) {
            stats = result.statsOrNull(Directions.Both, PlTypes.Money, SampleTypes.OutOfSample);
        }
        if (stats == null) return Double.NaN;
        return stats.getDouble(metric, Double.NaN);
    }

    /**
     * Pulls the two axis values out of a key like "WF: 6 runs : 28 % OOS".
     * Regex based so extra spacing or wording changes do not break it.
     */
    static int[] parseRunsAndOos(String key) {
        Matcher runs = RUNS_RE.matcher(key);
        Matcher oos  = OOS_RE.matcher(key);
        if (!runs.find() || !oos.find()) return null;
        try {
            return new int[] { Integer.parseInt(runs.group(1)), Integer.parseInt(oos.group(1)) };
        } catch (Exception e) {
            return null;
        }
    }

    // ------------------------------------------------------------------
    //  Reporting
    // ------------------------------------------------------------------

    private static String fmt(double v) {
        if (Double.isNaN(v)) return "n/a";
        if (Double.isInfinite(v)) return "inf";
        return String.format(Locale.US, "%.2f", v);
    }

    private static String fmtRatio(double v) {
        if (Double.isNaN(v)) return "n/a";
        if (Double.isInfinite(v)) return "inf";
        return String.format(Locale.US, "%.2f", v);
    }

    private static String dump(double[][] m, List<Integer> runsVals, List<Integer> oosVals,
                               String metric, Verdict v, Config c) {
        StringBuilder sb = new StringBuilder();
        sb.append("  ").append(metric).append(" surface (runs x OOS%), ")
          .append(v.validCells).append(" values, min ").append(fmt(v.globalMin))
          .append(" max ").append(fmt(v.globalMax))
          .append(", spread ").append(fmtRatio(v.globalRatio)).append("x\n");

        sb.append("        ");
        for (int col = 0; col < oosVals.size(); col++) {
            sb.append(String.format(Locale.US, "%9s", oosVals.get(col) + "%"));
        }
        sb.append('\n');

        for (int r = 0; r < m.length; r++) {
            sb.append(String.format(Locale.US, "%7s ", runsVals.get(r) + "r"));
            for (int col = 0; col < m[r].length; col++) {
                boolean inZone = v.bestRow >= 0
                              && r   >= v.bestRow && r   < v.bestRow + c.zone
                              && col >= v.bestCol && col < v.bestCol + c.zone;
                String cell = Double.isNaN(m[r][col]) ? "-" : fmt(m[r][col]);
                sb.append(String.format(Locale.US, "%9s", inZone ? "[" + cell + "]" : cell));
            }
            sb.append('\n');
        }

        if (v.bestRow >= 0) {
            sb.append("  flattest qualifying ").append(c.zone).append('x').append(c.zone)
              .append(" zone marked [] -> spread ").append(fmtRatio(v.bestZoneRatio))
              .append("x, mean ").append(fmt(v.bestZoneMean)).append('\n');
        } else {
            sb.append("  no complete ").append(c.zone).append('x').append(c.zone)
              .append(" zone passed the quality check\n");
        }
        return sb.toString();
    }

    @Override
    public ArrayList<ResultsGroup> processDatabank(String project, String task,
            String databankName, ArrayList<ResultsGroup> databankRG) throws Exception {
        return databankRG;
    }
}
