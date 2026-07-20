package SQ.Columns.Databanks;

import com.strategyquant.lib.L;
import com.strategyquant.lib.SettingsMap;
import com.strategyquant.tradinglib.DatabankColumn;
import com.strategyquant.tradinglib.Order;
import com.strategyquant.tradinglib.OrdersList;
import com.strategyquant.tradinglib.SQStats;
import com.strategyquant.tradinglib.StatsTypeCombination;
import com.strategyquant.tradinglib.ValueTypes;

/**
 * EdgeRatioNoAtr - MFE/MAE edge ratio built only from data a Walk-Forward
 * Matrix sub-result actually carries.
 *
 *     sum(|order.MFE|) / sum(|order.MAE|)      both in money
 *
 * Same idea as SQX's native Edge Ratio - how far the average trade runs in your
 * favour versus against you before it closes - without the two inputs that go
 * missing in a WFM. Higher is better; above 1 the favourable excursion wins.
 *
 * WHY THIS EXISTS
 *
 * Measured on real WFM sub-results, of the fields the native column needs:
 *
 *     order.MAE / order.MFE          present on 100% of trades
 *     order.PipsMAE / order.PipsMFE  present on 0%
 *     order.ATROnOpen                present on 0%
 *
 * ATROnOpen is the one field OrdersList.readAdditionalData() restores, and that
 * optional block is not carried by WF sub-results, so it is always absent there.
 * The pips excursions are a different story: loadOrderFormat10() does restore
 * them, they simply come back as 0 in a WF sub-result, which is why
 * OrderPLComputer's "derive MAE in money from PipsMAE" step is skipped and the
 * stored money values survive intact. Either way only the money fields can be
 * relied on here.
 *
 * The native EdgeRatioInPips needs the two that are missing. It divides by
 * ATROnOpen, so with that field at its default of 0 the arithmetic collapses
 * without complaining:
 *
 *     PipsMAE / 0        -> Infinity
 *     sum of Infinity    -> Infinity
 *     safeDivide(Inf,Inf)-> NaN
 *     round2(NaN)        -> 0.0     because Math.round(NaN) is 0 in Java
 *
 * So it reads 0 in every cell of a WFM, which looks like a measurement rather
 * than missing data.
 *
 * WHY MONEY IS FINE HERE
 *
 * This is a ratio of two excursions, so the unit largely cancels: with a fixed
 * position size the money ratio equals the pips ratio exactly. Where they
 * differ is money management - with variable position sizing the money ratio
 * weights bigger positions more heavily, while a pips ratio weights every trade
 * equally. For comparing runs of one strategy across a WFM that is not a
 * problem, and it is the only thing computable there at all.
 *
 * DELIBERATELY NOT ADAPTIVE
 *
 * It always uses the money fields, whatever the "Result in" selector says, and
 * regardless of whether the ATR happens to be available. One input, one scale,
 * no silent switching - and in particular it never returns a quiet 0 just
 * because Pips was selected and the pips fields are empty.
 *
 * Use it in the Walk-Forward Matrix, where the native column cannot work, and
 * keep the native Edge Ratio everywhere the ATR is intact - it is the better
 * measure when it can be computed, being volatility-normalised and therefore
 * comparable across symbols.
 *
 * Deploy to: <SQX>/user/extend/Snippets/SQ/Columns/Databanks/EdgeRatioNoAtr.java
 */
public class EdgeRatioNoAtr extends DatabankColumn {

    public EdgeRatioNoAtr() {
        super(L.t("Edge Ratio (no ATR)"), DatabankColumn.Decimal2, ValueTypes.Maximize, 0, 0, 50);

        setTooltip(L.t("MFE/MAE edge ratio from the money excursions, without the ATR "
                     + "normalisation. Works in the Walk-Forward Matrix, where the native "
                     + "Edge Ratio reads 0 because neither the ATR at entry nor the pips "
                     + "excursions are stored per run."));
    }

    // ------------------------------------------------------------------

    @Override
    public double compute(SQStats stats, StatsTypeCombination combination, OrdersList ordersList,
                          SettingsMap settings, SQStats statsLong, SQStats statsShort)
            throws Exception {

        double sumMAE = 0;
        double sumMFE = 0;

        for (int i = 0; i < ordersList.size(); i++) {
            Order order = ordersList.get(i);

            // deposits and withdrawals are not trades
            if (order.isBalanceOrder() || !order.isRealOrder()) continue;

            // Magnitudes only. MAE and MFE are excursion sizes, and their sign
            // convention does not matter to a ratio of the two - taking abs
            // means a negative-signed MAE cannot make the total non-positive
            // and trip the guard in ratio().
            double mae = Math.abs(order.MAE);
            double mfe = Math.abs(order.MFE);

            // one unusable trade must not poison the totals, which is the
            // failure mode that makes the native column collapse to 0
            if (!isUsable(mae) || !isUsable(mfe)) continue;

            sumMAE += mae;
            sumMFE += mfe;
        }

        return round2(ratio(sumMFE, sumMAE));
    }

    // ------------------------------------------------------------------

    /** Rejects NaN and the infinities a bad divide would otherwise spread. */
    static boolean isUsable(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    /**
     * MFE over MAE, guarding the degenerate cases.
     *
     * With no adverse excursion at all there is no ratio to report, so 0 is
     * returned rather than an infinity that would dominate every sort and
     * chart.
     */
    static double ratio(double sumMFE, double sumMAE) {
        if (sumMAE <= 0) return 0;
        double r = sumMFE / sumMAE;
        return isUsable(r) ? r : 0;
    }
}
