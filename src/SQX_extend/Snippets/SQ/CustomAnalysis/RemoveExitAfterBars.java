package SQ.CustomAnalysis;

import com.strategyquant.tradinglib.CustomAnalysisMethod;
import com.strategyquant.tradinglib.ResultsGroup;
import org.jdom2.Attribute;
import org.jdom2.Element;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * RemoveExitAfterBars - disables the Exit After Bars block on every strategy of
 * a databank, in place, without calling an external script.
 *
 * Replaces the old sqx_tool.py "remove_eab" step, which had to be invoked
 * through a CallExternalScript task that wrote patched .sqx files to a folder
 * and read them back with LoadFromFiles. This does the same job inside SQX, so
 * the extra round trip through disk disappears.
 *
 * HOW IT WORKS
 *
 * The Exit After Bars building block is a Param on each entry block:
 *
 *   <Param key="#ExitAfterBars.ExitAfterBars#" type="int" minValue="0"
 *          defaultValue="0" ...>12</Param>
 *
 * Its value is the number of bars after which the trade is force-closed, and
 * the block itself declares minValue="0" / defaultValue="0" - so 0 is the
 * block's own "do nothing" value. Setting it to 0 disables the exit while
 * leaving the strategy XML structurally intact, which is safer than deleting
 * the Param outright: nothing else in the file can end up pointing at a node
 * that no longer exists.
 *
 * The Param comes in two shapes and both are handled:
 *
 *   - a literal value       -> the text is replaced with 0
 *   - a variable reference  -> carries variable="true" and its text is the id
 *                              of a <variable> entry. The attribute is dropped
 *                              so the Param becomes a plain literal 0, leaving
 *                              the shared <variable> untouched for whatever
 *                              else may reference it.
 *
 * INPUT ARGUMENTS (optional)
 *
 *   (empty)   disable Exit After Bars   (same as EAB=0)
 *   EAB=0     disable Exit After Bars
 *   NoEAB     disable Exit After Bars
 *   EAB=12    force Exit After Bars to 12 bars on every strategy
 *   12        same, bare number
 *
 * Other comma separated tokens (SL=, TP=, ...) are ignored, so the argument
 * string can be shared with CAAddFixedSLTP without editing it.
 *
 * NOTE: like any strategy-XML edit, this does not re-run the backtest. The
 * databank metrics still describe the strategies as they were until a Retest
 * task runs after this one.
 *
 * Deploy to: <SQX>/user/extend/Snippets/SQ/CustomAnalysis/RemoveExitAfterBars.java
 */
public class RemoveExitAfterBars extends CustomAnalysisMethod {

    public static final Logger Log = LoggerFactory.getLogger("RemoveExitAfterBars");

    private static final String EXITAFTERBARS_PARAM_KEY = "#ExitAfterBars.ExitAfterBars#";
    private static final int    DEFAULT_BARS            = 0;

    public RemoveExitAfterBars() {
        super("RemoveExitAfterBars", TYPE_PROCESS_DATABANK);
    }

    @Override
    public boolean filterStrategy(String project, String task, String databankName,
                                  ResultsGroup rg) throws Exception {
        return true;
    }

    // ------------------------------------------------------------------
    //  Argument parsing
    // ------------------------------------------------------------------

    /**
     * Reads the target bar count out of the input args.
     *
     * Accepts "EAB=<n>", "NoEAB", a bare "<n>", or nothing at all. Tokens
     * belonging to other custom analyses are skipped rather than rejected.
     */
    static int parseBars(String raw) {
        if (raw == null) return DEFAULT_BARS;

        String[] tokens = raw.split(",");
        for (int i = 0; i < tokens.length; i++) {
            String token = tokens[i].trim();
            if (token.isEmpty()) continue;

            if (token.equalsIgnoreCase("NoEAB")) return 0;

            int eq = token.indexOf('=');
            if (eq >= 0) {
                String key = token.substring(0, eq).trim();
                if (!key.equalsIgnoreCase("EAB")) continue;   // belongs to someone else
                token = token.substring(eq + 1).trim();
            }

            try {
                int bars = (int) Double.parseDouble(token);
                return bars < 0 ? 0 : bars;
            } catch (NumberFormatException e) {
                Log.warn("RemoveExitAfterBars: ignoring unreadable argument [{}]", tokens[i].trim());
            }
        }
        return DEFAULT_BARS;
    }

    // ------------------------------------------------------------------
    //  Strategy XML surgery
    // ------------------------------------------------------------------

    /** Collects every &lt;Param&gt; with the given key, at any depth. */
    static void findParamsByKey(Element el, String key, List<Element> result) {
        if (el == null) return;
        if ("Param".equals(el.getName()) && key.equals(el.getAttributeValue("key"))) {
            result.add(el);
        }
        List<Element> children = el.getChildren();
        for (int i = 0; i < children.size(); i++) {
            findParamsByKey(children.get(i), key, result);
        }
    }

    /**
     * Pins every Exit After Bars Param to a literal value.
     *
     * @return {params found, params actually changed}
     */
    static int[] applyExitAfterBars(Element strategy, int bars) {
        List<Element> params = new ArrayList<Element>();
        findParamsByKey(strategy, EXITAFTERBARS_PARAM_KEY, params);

        String wanted = String.valueOf(bars);
        int changed = 0;

        for (int i = 0; i < params.size(); i++) {
            Element param = params.get(i);

            Attribute variable = param.getAttribute("variable");
            boolean wasReference = variable != null;
            boolean sameValue = wanted.equals(param.getTextTrim());

            if (wasReference) param.removeAttribute("variable");
            param.setText(wanted);

            if (wasReference || !sameValue) changed++;
        }
        return new int[] { params.size(), changed };
    }

    // ------------------------------------------------------------------
    //  SQX entry point
    // ------------------------------------------------------------------

    @Override
    public ArrayList<ResultsGroup> processDatabank(String project, String task,
            String databankName, ArrayList<ResultsGroup> databankRG) throws Exception {

        int bars = parseBars(getInputArgs());

        Log.info("=== RemoveExitAfterBars: {} strategies, setting Exit After Bars to {} ===",
                 databankRG.size(), bars == 0 ? "0 (disabled)" : String.valueOf(bars));

        int modified = 0, untouched = 0, skipped = 0, totalParams = 0;

        for (int i = 0; i < databankRG.size(); i++) {
            ResultsGroup rg = databankRG.get(i);
            try {
                Element strategy = rg.getStrategyXml();
                if (strategy == null) {
                    skipped++;
                    continue;
                }

                // Work on a clone so a failure part way through cannot leave the
                // strategy half rewritten.
                Element clone = strategy.clone();
                int[] result = applyExitAfterBars(clone, bars);
                totalParams += result[0];

                if (result[1] > 0) {
                    rg.setStrategyXml(clone);
                    modified++;
                    Log.debug("Exit After Bars -> {} on {} ({} of {} blocks changed)",
                              bars, rg.getName(), result[1], result[0]);
                } else {
                    untouched++;
                    if (result[0] == 0) {
                        Log.debug("No Exit After Bars block in {}", rg.getName());
                    }
                }

            } catch (Exception e) {
                Log.error("Error processing strategy '" + rg.getName() + "'", e);
                skipped++;
            }
        }

        Log.info("RemoveExitAfterBars finished. Modified: {}, already correct: {}, "
               + "skipped: {}, Exit After Bars blocks seen: {}",
                 modified, untouched, skipped, totalParams);

        setProjectLog("RemoveExitAfterBars: set Exit After Bars to " + bars
                    + " on " + modified + " of " + databankRG.size() + " strategies");

        return databankRG;
    }
}
