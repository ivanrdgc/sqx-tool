package SQ.CustomAnalysis;

import com.strategyquant.lib.*;
import com.strategyquant.datalib.*;
import com.strategyquant.tradinglib.*;

import SQ.Columns.Databanks.EntryIndicators;
import SQ.Columns.Databanks.PriceIndicators;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * SelectBestByIndicatorGroup - keeps the best strategy of each unique indicator
 * combination, where "unique" means BOTH the Entry indicators AND the Price
 * indicators match.
 *
 * Two strategies are considered duplicates only when they use the same entry
 * indicator set *and* the same price indicator set. Of each such group the one
 * with the highest fitness survives; the rest are dropped.
 *
 * This is a stricter grouping than SelectBestByEntryGroupArgs, which keys on the
 * Entry indicators alone. Those two columns are complementary in SQX: entry
 * indicators are the blocks used in the entry *conditions*, price indicators are
 * the blocks used in the entry *price levels* (Enter at Stop / Limit). Grouping
 * on entry alone therefore treats two strategies as the same edge even when one
 * enters at a Bollinger band and the other at a moving average, and throws one
 * of them away. Keying on both keeps them apart.
 *
 * Both keys come from SQX's own EntryIndicators / PriceIndicators databank
 * columns, so the grouping matches exactly what those columns display. Each
 * builds its value through a TreeSet, meaning the string is already sorted and
 * de-duplicated - "MA,RSI" regardless of the order the blocks appear in the
 * strategy - so equal indicator sets always produce an equal key.
 *
 * USAGE
 *   - Add as a "Full databank analysis" task in a Custom Project
 *   - Set the Source databank (strategies to evaluate)
 *   - Set the Target databank (winners are copied here)
 *   - Turn OFF "Filter by results of custom analysis"
 *
 * INPUT ARGUMENTS (optional): Criterion,N,Sample
 *
 *   (empty)            best 1 per group by Fitness          <- the default
 *   Fitness,2          best 2 per group by Fitness
 *   RetDD              best 1 per group by Ret/DD, full sample
 *   RetDD,3,OOS        best 3 per group by Ret/DD, out of sample
 *   NetProfit,1,IS     best 1 per group by Net Profit, in sample
 *
 *   Criterion: Fitness (default), RetDD, Sharpe, ProfitFactor, NetProfit,
 *              Calmar, WinRate, Drawdown (smaller is better)
 *   N        : how many to keep per group, 1..10 (default 1)
 *   Sample   : Full (default), IS, OOS. Ignored when Criterion is Fitness,
 *              which is a single value per strategy.
 *
 * Strategies whose indicators cannot be determined (either column returns
 * "N/A", e.g. the strategy XML is unreadable) are never grouped with anything
 * and always survive. Dropping a strategy we failed to identify would be the
 * one unrecoverable mistake this filter could make.
 *
 * Output keeps the databank's original ordering, and ties on the criterion are
 * broken by that same original order, so runs are reproducible.
 *
 * Deploy to: <SQX>/user/extend/Snippets/SQ/CustomAnalysis/SelectBestByIndicatorGroup.java
 */
public class SelectBestByIndicatorGroup extends CustomAnalysisMethod {

    public static final Logger Log = LoggerFactory.getLogger("SelectBestByIndicatorGroup");

    private static final String DEFAULT_CRITERION = "Fitness";
    private static final int    DEFAULT_TOP_N     = 1;
    private static final int    MAX_TOP_N         = 10;

    /** Criteria where a SMALLER value is better. */
    private static final Set<String> MINIMIZE_CRITERIA = new HashSet<>(
        java.util.Arrays.asList("drawdown")
    );

    /** Value both indicator columns return when they cannot read the strategy. */
    private static final String NOT_AVAILABLE = DatabankColumn.NOT_AVAILABLE;

    public SelectBestByIndicatorGroup() {
        super("SelectBestByIndicatorGroup", TYPE_PROCESS_DATABANK);
    }

    @Override
    public boolean filterStrategy(String project, String task, String databankName,
                                  ResultsGroup rg) throws Exception {
        return true;
    }

    // ------------------------------------------------------------------
    //  Configuration
    // ------------------------------------------------------------------

    static class Config {
        String criterion  = DEFAULT_CRITERION;
        int    topN       = DEFAULT_TOP_N;
        byte   sampleType = SampleTypes.FullSample;
        String sampleLabel = "Full";
        boolean minimize  = false;
    }

    static Config parseArgs(String raw) {
        Config c = new Config();
        if (raw == null) return finish(c);

        String args = raw.trim();
        if (args.isEmpty()) return finish(c);

        String[] parts = args.split(",");

        if (parts.length >= 1 && !parts[0].trim().isEmpty()) {
            c.criterion = parts[0].trim();
        }

        if (parts.length >= 2 && !parts[1].trim().isEmpty()) {
            try {
                c.topN = Integer.parseInt(parts[1].trim());
            } catch (NumberFormatException e) {
                Log.warn("Invalid N [{}], using {}", parts[1].trim(), DEFAULT_TOP_N);
                c.topN = DEFAULT_TOP_N;
            }
        }

        if (parts.length >= 3) {
            String sample = parts[2].trim().toUpperCase(Locale.US);
            if (sample.equals("IS")) {
                c.sampleType = SampleTypes.InSample;
                c.sampleLabel = "IS";
            } else if (sample.equals("OOS")) {
                c.sampleType = SampleTypes.OutOfSample;
                c.sampleLabel = "OOS";
            }
        }

        return finish(c);
    }

    private static Config finish(Config c) {
        if (c.topN < 1)         c.topN = 1;
        if (c.topN > MAX_TOP_N) c.topN = MAX_TOP_N;
        c.minimize = MINIMIZE_CRITERIA.contains(c.criterion.toLowerCase(Locale.US));
        return c;
    }

    // ------------------------------------------------------------------
    //  Grouping and selection  (pure logic, unit testable)
    // ------------------------------------------------------------------

    /** One strategy reduced to what the selection actually needs. */
    static class Scored {
        final int    index;     // position in the source databank
        final String key;       // entry indicators + price indicators
        final double score;
        final String name;

        Scored(int index, String key, double score, String name) {
            this.index = index;
            this.key   = key;
            this.score = score;
            this.name  = name;
        }
    }

    /**
     * Groups by key and returns the source indices of the winners, in the
     * original databank order.
     *
     * Sorting is stable and NaN scores always rank last, so a strategy whose
     * criterion could not be computed never displaces one that has a real score.
     */
    static List<Integer> selectWinners(List<Scored> items, int topN, boolean minimize) {
        Map<String, List<Scored>> groups = new LinkedHashMap<>();
        for (Scored s : items) {
            List<Scored> group = groups.get(s.key);
            if (group == null) {
                group = new ArrayList<>();
                groups.put(s.key, group);
            }
            group.add(s);
        }

        Comparator<Scored> best = (a, b) -> {
            boolean aNaN = Double.isNaN(a.score);
            boolean bNaN = Double.isNaN(b.score);
            if (aNaN && bNaN) return 0;
            if (aNaN) return 1;
            if (bNaN) return -1;
            return minimize ? Double.compare(a.score, b.score)
                            : Double.compare(b.score, a.score);
        };

        Set<Integer> winners = new HashSet<>();
        for (List<Scored> group : groups.values()) {
            List<Scored> sorted = new ArrayList<>(group);
            Collections.sort(sorted, best);          // stable: ties keep databank order
            int limit = Math.min(topN, sorted.size());
            for (int i = 0; i < limit; i++) {
                winners.add(sorted.get(i).index);
            }
        }

        List<Integer> ordered = new ArrayList<>(winners);
        Collections.sort(ordered);
        return ordered;
    }

    /** Number of distinct groups in the given items. */
    static int countGroups(List<Scored> items) {
        Set<String> keys = new HashSet<>();
        for (Scored s : items) keys.add(s.key);
        return keys.size();
    }

    /**
     * Builds the grouping key from the two indicator sets.
     *
     * Returns null when either side is unavailable, which the caller turns into
     * a one-off key so the strategy is never treated as anyone's duplicate.
     * An EMPTY string is not the same as unavailable - a strategy that simply
     * has no price indicators legitimately groups with others that have none.
     */
    static String buildKey(String entryIndicators, String priceIndicators) {
        if (entryIndicators == null || priceIndicators == null) return null;
        String entry = entryIndicators.trim();
        String price = priceIndicators.trim();
        if (entry.equals(NOT_AVAILABLE) || price.equals(NOT_AVAILABLE)) return null;
        if (entry.isEmpty() && price.isEmpty()) return null;
        return entry + " | " + price;
    }

    // ------------------------------------------------------------------
    //  SQX entry point
    // ------------------------------------------------------------------

    @Override
    public ArrayList<ResultsGroup> processDatabank(String project, String task,
            String databankName, ArrayList<ResultsGroup> databankRG) throws Exception {

        Config c = parseArgs(getInputArgs());

        Log.info("=== SelectBestByIndicatorGroup: {} strategies | criterion {} | top {} | sample {} ===",
                 databankRG.size(), c.criterion, c.topN,
                 c.criterion.equalsIgnoreCase("Fitness") ? "n/a" : c.sampleLabel);

        EntryIndicators entryCol = new EntryIndicators();
        PriceIndicators priceCol = new PriceIndicators();

        List<Scored> items = new ArrayList<>();
        int unresolved = 0;

        for (int i = 0; i < databankRG.size(); i++) {
            ResultsGroup rg = databankRG.get(i);
            String name = rg.getName();
            try {
                String entry = entryCol.getValue(rg, null, (byte) 0, (byte) 0, (byte) 0);
                String price = priceCol.getValue(rg, null, (byte) 0, (byte) 0, (byte) 0);

                String key = buildKey(entry, price);
                if (key == null) {
                    // Never group something we could not identify.
                    key = "##unresolved#" + i;
                    unresolved++;
                    Log.warn("  {} | indicators unavailable (entry=[{}] price=[{}]) - kept",
                             name, entry, price);
                }

                double score = calcScore(rg, c);
                items.add(new Scored(i, key, score, name));

                Log.debug("  {} | key=[{}] | {}={}", name, key, c.criterion, score);

            } catch (Exception e) {
                // Keep it rather than silently discard it.
                Log.warn("  ERROR on {}: {} - kept", name, e.getMessage());
                items.add(new Scored(i, "##error#" + i, Double.NaN, name));
                unresolved++;
            }
        }

        int groups = countGroups(items);
        List<Integer> winnerIndices = selectWinners(items, c.topN, c.minimize);

        ArrayList<ResultsGroup> winners = new ArrayList<>();
        for (int i = 0; i < winnerIndices.size(); i++) {
            winners.add(databankRG.get(winnerIndices.get(i)));
        }

        int removed = databankRG.size() - winners.size();
        Log.info("=== Result: {} kept, {} removed as duplicates, {} unique indicator groups ===",
                 winners.size(), removed, groups);
        if (unresolved > 0) {
            Log.info("    {} strategies could not be identified and were kept untouched", unresolved);
        }

        setProjectLog("SelectBestByIndicatorGroup: kept " + winners.size() + " of "
                    + databankRG.size() + " (" + groups + " unique Entry+Price groups)");

        return winners;
    }

    // ------------------------------------------------------------------
    //  Scoring
    // ------------------------------------------------------------------

    private double calcScore(ResultsGroup rg, Config c) throws Exception {
        if (c.criterion.equalsIgnoreCase("Fitness")) {
            return rg.getFitness();
        }

        String mainKey = getMainResultKey(rg);
        if (mainKey == null) return Double.NaN;

        Result result = rg.subResult(mainKey);
        if (result == null) return Double.NaN;

        SQStats stats = result.statsOrNull(Directions.Both, PlTypes.Money, c.sampleType);
        if (stats == null) return Double.NaN;

        if (stats.getInt("NumberOfTrades") == 0) return Double.NaN;

        String criterion = c.criterion.toLowerCase(Locale.US);
        if (criterion.equals("sharpe"))       return stats.getDouble("SharpeRatio",  Double.NaN);
        if (criterion.equals("profitfactor")) return stats.getDouble("ProfitFactor", Double.NaN);
        if (criterion.equals("netprofit"))    return stats.getDouble("NetProfit",    Double.NaN);
        if (criterion.equals("calmar"))       return stats.getDouble("CalmarRatio",  Double.NaN);
        if (criterion.equals("winrate"))      return stats.getDouble("PercProfitableTrades", Double.NaN);
        if (criterion.equals("drawdown"))     return stats.getDouble("Drawdown",     Double.NaN);
        return stats.getDouble("ReturnDDRatio", Double.NaN);   // RetDD and anything unknown
    }

    /** Main backtest result key, skipping crosscheck / walk-forward / portfolio entries. */
    private String getMainResultKey(ResultsGroup rg) throws Exception {
        String firstKey = null;
        for (Object key : rg.getResultKeys()) {
            String k = (String) key;
            if (firstKey == null) firstKey = k;
            if (!k.contains("CrossCheck") && !k.contains("WF") && !k.contains("Portfolio")) {
                return k;
            }
        }
        return firstKey;
    }
}
