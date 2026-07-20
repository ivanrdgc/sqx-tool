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
 * ReturnOnRisk - realised profit per unit of money actually put at risk.
 *
 *     A = sum of net P/L of every trade          (money, costs included)
 *     B = sum of |MAE| + |commission and swap|   (money)
 *
 *     value = A / B
 *
 * "For every dollar that went underwater or went to the broker, how many dollars
 * did I keep?". Higher is better. 0.20 means you cleared 20 cents per dollar of
 * risk you actually lived through.
 *
 * WHY THERE IS NO DIVISION BY YEARS
 *
 * A and B are both cumulative sums over trades, so a longer backtest grows both
 * of them at roughly the same rate and the ratio converges on
 *
 *     mean(profit per trade) / mean(risk per trade)
 *
 * which is already independent of how long the backtest ran. Dividing by the
 * number of years would not remove a length dependence, it would introduce one -
 * a 20 year backtest would score half of a 10 year backtest of identical
 * quality.
 *
 * The rule of thumb: annualise when the risk denominator is a peak that does not
 * grow with time, such as max drawdown - that is why CAGR/Max DD % exists.
 * Summed MAE does grow with time, so it needs no such correction.
 *
 * WHAT IT DOES NOT MEASURE
 *
 * Speed. Two strategies with the same value, one trading 10 times a year and one
 * 1000 times, are equally efficient per dollar risked but earn wildly different
 * absolute amounts. Read trade frequency or profit per year alongside this, not
 * folded into it.
 *
 * WHY NOT ONE OF THE BUILT-INS
 *
 * SQX has no column that uses summed MAE as the denominator. The near relatives
 * all measure risk differently:
 *
 *     Ret/DD Ratio     NetProfit / max drawdown   - risk = worst equity dip
 *     Profit factor    gross profit / gross loss  - risk = losses actually taken
 *     CAGR/Max DD %    compounded                 - assumes reinvestment
 *
 * Drawdown is one worst moment and gross loss only counts trades that ended
 * badly. Summed MAE counts the heat of every trade, including winners that were
 * deeply underwater before they recovered, which is the exposure you actually
 * lived through. And because this is a plain ratio of sums it stays linear - no
 * compounding, matching fixed-amount risk sizing.
 *
 * COSTS ARE COUNTED ONCE, ON EACH SIDE
 *
 * When a project has "AddCommissionSwapToPL" enabled SQX folds commission, swap
 * and slippage into Order.PL as the orders are computed, and records that it did
 * so in Order.CommSwapApplied. So Order.PL is normally already net, and adding
 * CommSwap to it again would count the costs twice. This reads the flag and only
 * applies the costs itself when SQX has not.
 *
 * On the risk side the costs are added as magnitudes, on top of |MAE|: money
 * paid to the broker is money put at risk no matter how the trade went.
 *
 * ALWAYS IN MONEY
 *
 * Like EdgeRatioNoAtr this ignores the "Result in" selector and always uses the
 * money fields. Every field it touches - PL, CommSwap, CommSwapApplied, MAE - is
 * restored by the main order load format, so this works in a Walk-Forward Matrix
 * sub-result too, unlike anything built on ATROnOpen.
 *
 * Deploy to: <SQX>/user/extend/Snippets/SQ/Columns/Databanks/ReturnOnRisk.java
 */
public class ReturnOnRisk extends DatabankColumn {

    public ReturnOnRisk() {
        super(L.t("Return/Risk"), DatabankColumn.Decimal4, ValueTypes.Maximize, 0, 0, 1);

        setTooltip(L.t("Net profit divided by the total money put at risk (sum of MAE "
                     + "plus commission and swap). Independent of backtest length, "
                     + "linear rather than compounded, always computed in money."));
    }

    // ------------------------------------------------------------------

    @Override
    public double compute(SQStats stats, StatsTypeCombination combination, OrdersList ordersList,
                          SettingsMap settings, SQStats statsLong, SQStats statsShort)
            throws Exception {

        double profit = 0; // A
        double risk = 0;   // B

        for (int i = 0; i < ordersList.size(); i++) {
            Order order = ordersList.get(i);

            // deposits and withdrawals are not trades
            if (order.isBalanceOrder() || !order.isRealOrder()) continue;

            double costs = order.CommSwap;
            double pl = order.PL;

            // only apply the costs ourselves if SQX has not already done it
            if (!order.CommSwapApplied) {
                pl = pl + costs - order.SlippageInMoney;
            }

            // Magnitudes. MAE and CommSwap are both stored signed, and a sign
            // flip must not be able to cancel risk out of the denominator.
            double tradeRisk = Math.abs(order.MAE) + Math.abs(costs);

            // one unusable trade must not poison the totals
            if (!isUsable(pl) || !isUsable(tradeRisk)) continue;

            profit += pl;
            risk += tradeRisk;
        }

        return round4(ratio(profit, risk));
    }

    // ------------------------------------------------------------------

    /** Rejects NaN and the infinities a bad divide would otherwise spread. */
    static boolean isUsable(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    /**
     * A / B, guarding the degenerate cases.
     *
     * With no risk recorded at all there is nothing to divide by, so 0 is
     * returned rather than an infinity that would top every sort.
     */
    static double ratio(double profit, double risk) {
        if (risk <= 0) return 0;
        double value = profit / risk;
        return isUsable(value) ? value : 0;
    }
}
