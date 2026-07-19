package SQ.CustomAnalysis;

import com.strategyquant.tradinglib.CustomAnalysisMethod;
import com.strategyquant.tradinglib.ResultsGroup;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

/**
 * RenameStrategies - gives strategies a readable name inside the databank.
 *
 * Two independent edits, applied in this order:
 *
 *   1. the leading "Strategy" word is replaced with the input text
 *   2. the word "Improved" is dropped from the separator SQX inserts
 *
 * so a project that renames at E-Build and improves at S-Build ends up with:
 *
 *   E-Build   Strategy 12345                        -> XAUUSD H1 Long 12345
 *   S-Build   XAUUSD H1 Long 12345 - Improved 678   -> XAUUSD H1 Long 12345 - 678
 *
 * Step 2 runs whether or not any input text was given, so the same snippet
 * works at both stages: pass the prefix at E-Build, leave it empty at S-Build
 * where the name is already correct and only "Improved" needs to go.
 *
 * INPUT ARGUMENTS (optional)
 *
 *   XAUUSD H1 Long    replace the leading "Strategy" with this text,
 *                     and drop "Improved"
 *   (empty)           only drop "Improved", leave the rest of the name alone
 *
 * Running it twice is safe. Once a name no longer starts with "Strategy " and
 * contains no "Improved", both steps are no-ops and nothing is written.
 *
 * PER STRATEGY OR FULL DATABANK
 *
 * Registered as TYPE_BOTH, so it can be selected in either slot of the Custom
 * Analysis task - "per strategy" or "full databank" - whichever fits the task
 * being set up. The result is identical: SQX hands the same ResultsGroup
 * objects to both hooks and keeps using them afterwards, so a rename applied in
 * either place sticks.
 *
 * In per-strategy mode SQX drops any strategy whose filterStrategy returns
 * false, so this one always returns true. It renames, it never filters.
 *
 * Replaces sqx_tool.py's rename_files, which did the same job to the exported
 * files through a CallExternalScript task. Renaming the strategy itself means
 * the name is right everywhere it shows up - the databank, the exported .sqx
 * and .mq5 files, and the generated source.
 *
 * Deploy to: <SQX>/user/extend/Snippets/SQ/CustomAnalysis/RenameStrategies.java
 */
public class RenameStrategies extends CustomAnalysisMethod {

    public static final Logger Log = LoggerFactory.getLogger("RenameStrategies");

    /** SQX's default strategy name prefix, including the trailing space. */
    private static final String DEFAULT_PREFIX = "Strategy ";

    /** How SQX joins an improved strategy onto the name of its parent. */
    private static final String IMPROVED_SEPARATOR   = " - Improved ";
    private static final String IMPROVED_REPLACEMENT = " - ";

    public RenameStrategies() {
        super("RenameStrategies", TYPE_BOTH);
    }

    /**
     * Per-strategy entry point.
     *
     * ALWAYS returns true. SQX removes a strategy from the databank when
     * filterStrategy returns false, and this method exists to rename, not to
     * filter - so even a failure has to leave the strategy in place.
     */
    @Override
    public boolean filterStrategy(String project, String task, String databankName,
                                  ResultsGroup rg) throws Exception {
        applyTo(rg, currentPrefix());
        return true;
    }

    // ------------------------------------------------------------------
    //  Naming  (pure logic, unit testable)
    // ------------------------------------------------------------------

    /**
     * Applies both edits to one strategy name.
     *
     * @param currentName the strategy's current name
     * @param prefix      text to put in place of the leading "Strategy",
     *                    or null/empty to only drop "Improved"
     * @return the new name, or null when nothing changed - which is what makes
     *         a second run a no-op
     */
    static String rename(String currentName, String prefix) {
        if (currentName == null) return null;

        String name = currentName.trim();
        String result = name;

        // 1. "Strategy 12345" -> "<prefix> 12345", keeping the id
        if (prefix != null && !prefix.isEmpty() && result.startsWith(DEFAULT_PREFIX)) {
            String rest = result.substring(DEFAULT_PREFIX.length()).trim();
            if (!rest.isEmpty()) {
                result = prefix + " " + rest;
            }
        }

        // 2. "... - Improved 678" -> "... - 678", every occurrence, always
        result = result.replace(IMPROVED_SEPARATOR, IMPROVED_REPLACEMENT);

        return result.equals(name) ? null : result;
    }

    // ------------------------------------------------------------------
    //  SQX entry point
    // ------------------------------------------------------------------

    /** Full-databank entry point. */
    @Override
    public ArrayList<ResultsGroup> processDatabank(String project, String task,
            String databankName, ArrayList<ResultsGroup> databankRG) throws Exception {

        String prefix = currentPrefix();

        Log.info("=== RenameStrategies: {} strategies | {} ===", databankRG.size(),
                 prefix.isEmpty()
                     ? "no prefix given, only dropping \"Improved\""
                     : "prefix \"" + prefix + "\", dropping \"Improved\"");

        int renamed = 0;
        for (int i = 0; i < databankRG.size(); i++) {
            if (applyTo(databankRG.get(i), prefix)) renamed++;
        }

        Log.info("=== Result: {} renamed, {} already correct ===",
                 renamed, databankRG.size() - renamed);

        setProjectLog("RenameStrategies: renamed " + renamed + " of " + databankRG.size()
                    + " strategies");

        return databankRG;
    }

    // ------------------------------------------------------------------
    //  Shared by both entry points
    // ------------------------------------------------------------------

    private String currentPrefix() {
        return getInputArgs() != null ? getInputArgs().trim() : "";
    }

    /**
     * Renames one strategy in place.
     *
     * Never throws: a strategy that cannot be renamed is logged and left with
     * the name it had, which matters most in per-strategy mode where the
     * caller must not fail.
     *
     * @return true when the name actually changed
     */
    private boolean applyTo(ResultsGroup rg, String prefix) {
        String oldName = null;
        try {
            oldName = rg.getName();
            String newName = rename(oldName, prefix);
            if (newName == null) {
                Log.debug("  {} | left unchanged", oldName);
                return false;
            }
            rg.setName(newName);
            Log.debug("  {} -> {}", oldName, newName);
            return true;
        } catch (Exception e) {
            Log.warn("  ERROR renaming {}: {} - left unchanged", oldName, e.getMessage());
            return false;
        }
    }
}
