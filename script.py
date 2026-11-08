import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict
from typing import Any


# ==================================================
# Financial Summary
# ==================================================

@dataclass
class FinancialSummary:
    total_revenue: float
    cogs: float
    gross_profit: float
    gross_margin_pct: float
    operating_expenses: float
    operating_income_ebit: float
    operating_margin_pct: float
    break_even_revenue: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# ==================================================
# Financial Intelligence Engine
# ==================================================

class FinancialIntelligenceEngine:
    """
    يحسب قائمة الدخل، والتحليل الشهري،
    وربحية المنتجات، وهوامش المساهمة.
    """

    REQUIRED_TRANSACTION_COLUMNS = {
        "date",
        "category",
        "amount",
    }

    REQUIRED_PRODUCT_COLUMNS = {
        "product_name",
        "unit_price",
        "unit_variable_cost",
        "units_sold",
    }

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ):
        self.transactions = transactions_df.copy()
        self.products = products_df.copy()

        self._validate_columns()
        self._prepare_data()

    # ==================================================
    # Data Validation
    # ==================================================

    def _validate_columns(self) -> None:
        """التأكد من وجود الأعمدة المطلوبة."""

        missing_transaction_columns = (
            self.REQUIRED_TRANSACTION_COLUMNS
            - set(self.transactions.columns)
        )

        missing_product_columns = (
            self.REQUIRED_PRODUCT_COLUMNS
            - set(self.products.columns)
        )

        if missing_transaction_columns:
            raise ValueError(
                "Missing transaction columns: "
                f"{sorted(missing_transaction_columns)}"
            )

        if missing_product_columns:
            raise ValueError(
                "Missing product columns: "
                f"{sorted(missing_product_columns)}"
            )

    def _prepare_data(self) -> None:
        """تنظيف البيانات وتحويل أنواعها."""

        self.transactions["date"] = pd.to_datetime(
            self.transactions["date"],
            errors="coerce",
        )

        self.transactions["amount"] = pd.to_numeric(
            self.transactions["amount"],
            errors="coerce",
        )

        product_numeric_columns = [
            "unit_price",
            "unit_variable_cost",
            "units_sold",
        ]

        for column in product_numeric_columns:
            self.products[column] = pd.to_numeric(
                self.products[column],
                errors="coerce",
            )

        if self.transactions["date"].isna().any():
            raise ValueError(
                "Some transaction dates are invalid."
            )

        if self.transactions["amount"].isna().any():
            raise ValueError(
                "Some transaction amounts are invalid."
            )

        if (
            self.products[product_numeric_columns]
            .isna()
            .any()
            .any()
        ):
            raise ValueError(
                "Some product values are invalid."
            )

        if (
            self.products[product_numeric_columns] < 0
        ).any().any():
            raise ValueError(
                "Product values cannot be negative."
            )

    # ==================================================
    # Income Statement
    # ==================================================

    def generate_income_statement(
        self,
    ) -> FinancialSummary:
        """إنشاء ملخص قائمة الدخل."""

        grouped = (
            self.transactions
            .groupby("category")["amount"]
            .sum()
            .to_dict()
        )

        revenue = grouped.get("Revenue", 0.0)
        cogs = grouped.get("COGS", 0.0)

        operating_expenses = sum(
            amount
            for category, amount in grouped.items()
            if category.startswith("OpEx_")
        )

        gross_profit = revenue - cogs

        if revenue > 0:
            gross_margin_pct = (
                gross_profit / revenue * 100
            )
        else:
            gross_margin_pct = 0.0

        operating_income = (
            gross_profit - operating_expenses
        )

        if revenue > 0:
            operating_margin_pct = (
                operating_income / revenue * 100
            )

            contribution_margin_ratio = (
                gross_profit / revenue
            )
        else:
            operating_margin_pct = 0.0
            contribution_margin_ratio = 0.0

        if contribution_margin_ratio > 0:
            break_even_revenue = (
                operating_expenses
                / contribution_margin_ratio
            )
        else:
            break_even_revenue = float("inf")

        return FinancialSummary(
            total_revenue=round(revenue, 2),
            cogs=round(cogs, 2),
            gross_profit=round(gross_profit, 2),
            gross_margin_pct=round(
                gross_margin_pct,
                2,
            ),
            operating_expenses=round(
                operating_expenses,
                2,
            ),
            operating_income_ebit=round(
                operating_income,
                2,
            ),
            operating_margin_pct=round(
                operating_margin_pct,
                2,
            ),
            break_even_revenue=round(
                break_even_revenue,
                2,
            ),
        )

    # ==================================================
    # Monthly Revenue Analysis
    # ==================================================

    def analyze_monthly_revenue(
        self,
    ) -> pd.DataFrame:
        """
        تحليل الإيرادات والتكاليف والأرباح
        ونمو الإيرادات شهريًا.
        """

        df = self.transactions.copy()

        df["month"] = (
            df["date"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        monthly_data = (
            df.pivot_table(
                index="month",
                columns="category",
                values="amount",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )

        if "Revenue" not in monthly_data.columns:
            monthly_data["Revenue"] = 0.0

        if "COGS" not in monthly_data.columns:
            monthly_data["COGS"] = 0.0

        opex_columns = [
            column
            for column in monthly_data.columns
            if str(column).startswith("OpEx_")
        ]

        if opex_columns:
            monthly_data["operating_expenses"] = (
                monthly_data[opex_columns].sum(axis=1)
            )
        else:
            monthly_data["operating_expenses"] = 0.0

        monthly_data["gross_profit"] = (
            monthly_data["Revenue"]
            - monthly_data["COGS"]
        )

        monthly_data["operating_income_ebit"] = (
            monthly_data["gross_profit"]
            - monthly_data["operating_expenses"]
        )

        monthly_data["gross_margin_pct"] = 0.0
        monthly_data["operating_margin_pct"] = 0.0

        valid_revenue = monthly_data["Revenue"] > 0

        monthly_data.loc[
            valid_revenue,
            "gross_margin_pct",
        ] = (
            monthly_data.loc[
                valid_revenue,
                "gross_profit",
            ]
            / monthly_data.loc[
                valid_revenue,
                "Revenue",
            ]
            * 100
        )

        monthly_data.loc[
            valid_revenue,
            "operating_margin_pct",
        ] = (
            monthly_data.loc[
                valid_revenue,
                "operating_income_ebit",
            ]
            / monthly_data.loc[
                valid_revenue,
                "Revenue",
            ]
            * 100
        )

        monthly_data["revenue_growth_pct"] = (
            monthly_data["Revenue"]
            .pct_change(fill_method=None)
            .mul(100)
        )

        monthly_data["month_name"] = (
            monthly_data["month"]
            .dt.strftime("%Y-%m")
        )

        result_columns = [
            "month",
            "month_name",
            "Revenue",
            "COGS",
            "gross_profit",
            "gross_margin_pct",
            "operating_expenses",
            "operating_income_ebit",
            "operating_margin_pct",
            "revenue_growth_pct",
        ]

        monthly_data = monthly_data[result_columns]

        numeric_columns = [
            "Revenue",
            "COGS",
            "gross_profit",
            "gross_margin_pct",
            "operating_expenses",
            "operating_income_ebit",
            "operating_margin_pct",
            "revenue_growth_pct",
        ]

        monthly_data[numeric_columns] = (
            monthly_data[numeric_columns].round(2)
        )

        return (
            monthly_data
            .sort_values(by="month")
            .reset_index(drop=True)
        )

    # ==================================================
    # Product Analysis
    # ==================================================

    def analyze_product_profitability(
        self,
    ) -> pd.DataFrame:
        """تحليل ربحية كل منتج وتصنيفه."""

        df = self.products.copy()

        df["total_revenue"] = (
            df["units_sold"]
            * df["unit_price"]
        )

        df["total_variable_cost"] = (
            df["units_sold"]
            * df["unit_variable_cost"]
        )

        df["contribution_margin"] = (
            df["total_revenue"]
            - df["total_variable_cost"]
        )

        df["cm_percentage"] = 0.0

        valid_revenue = df["total_revenue"] > 0

        df.loc[
            valid_revenue,
            "cm_percentage",
        ] = (
            df.loc[
                valid_revenue,
                "contribution_margin",
            ]
            / df.loc[
                valid_revenue,
                "total_revenue",
            ]
            * 100
        )

        median_volume = df["units_sold"].median()
        median_margin = df["cm_percentage"].median()

        def classify_product(
            row: pd.Series,
        ) -> str:
            high_margin = (
                row["cm_percentage"] >= median_margin
            )

            high_volume = (
                row["units_sold"] >= median_volume
            )

            if high_margin and high_volume:
                return "Stars"

            if not high_margin and high_volume:
                return "Volume Drivers"

            if high_margin and not high_volume:
                return "Niche Opportunities"

            return "Underperformers"

        df["classification"] = df.apply(
            classify_product,
            axis=1,
        )

        numeric_columns = [
            "total_revenue",
            "total_variable_cost",
            "contribution_margin",
            "cm_percentage",
        ]

        df[numeric_columns] = (
            df[numeric_columns].round(2)
        )

        return (
            df.sort_values(
                by="contribution_margin",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    # ==================================================
    # Product Charts
    # ==================================================

    def create_product_charts(
        self,
        product_analysis: pd.DataFrame,
        save_path: str = "product_analysis.png",
    ) -> None:
        """
        إنشاء لوحة رسوم بيانية لتحليل المنتجات.
        """

        if product_analysis.empty:
            print(
                "No product data available for charts."
            )
            return

        df = product_analysis.copy()

        figure, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(16, 11),
        )

        figure.suptitle(
            "Product Profitability Analysis",
            fontsize=18,
            fontweight="bold",
        )

        # الرسم الأول: الإيرادات والتكاليف
        x_positions = list(range(len(df)))
        bar_width = 0.35

        revenue_positions = [
            position - bar_width / 2
            for position in x_positions
        ]

        cost_positions = [
            position + bar_width / 2
            for position in x_positions
        ]

        axes[0, 0].bar(
            revenue_positions,
            df["total_revenue"],
            width=bar_width,
            label="Revenue",
            color="#3498db",
        )

        axes[0, 0].bar(
            cost_positions,
            df["total_variable_cost"],
            width=bar_width,
            label="Variable Cost",
            color="#e74c3c",
        )

        axes[0, 0].set_title(
            "Revenue vs Variable Cost"
        )

        axes[0, 0].set_ylabel("Amount ($)")
        axes[0, 0].set_xticks(x_positions)

        axes[0, 0].set_xticklabels(
            df["product_name"],
            rotation=25,
            ha="right",
        )

        axes[0, 0].legend()

        axes[0, 0].grid(
            axis="y",
            linestyle="--",
            alpha=0.4,
        )

        # الرسم الثاني: هامش المساهمة
        margin_colors = []

        for value in df["cm_percentage"]:
            if value >= 30:
                margin_colors.append("#27ae60")
            elif value >= 10:
                margin_colors.append("#f39c12")
            else:
                margin_colors.append("#c0392b")

        axes[0, 1].barh(
            df["product_name"],
            df["cm_percentage"],
            color=margin_colors,
        )

        axes[0, 1].axvline(
            x=10,
            color="#c0392b",
            linestyle="--",
            label="Minimum Target: 10%",
        )

        axes[0, 1].set_title(
            "Contribution Margin Percentage"
        )

        axes[0, 1].set_xlabel(
            "Contribution Margin (%)"
        )

        axes[0, 1].legend()

        axes[0, 1].grid(
            axis="x",
            linestyle="--",
            alpha=0.4,
        )

        for index, value in enumerate(
            df["cm_percentage"]
        ):
            axes[0, 1].text(
                value + 1,
                index,
                f"{value:.1f}%",
                va="center",
            )

        # الرسم الثالث: مصفوفة المنتجات
        classification_colors = {
            "Stars": "#2ecc71",
            "Volume Drivers": "#3498db",
            "Niche Opportunities": "#f1c40f",
            "Underperformers": "#e74c3c",
        }

        for classification, group in df.groupby(
            "classification"
        ):
            bubble_sizes = (
                group["total_revenue"] / 150
            ).clip(lower=50)

            axes[1, 0].scatter(
                group["units_sold"],
                group["cm_percentage"],
                s=bubble_sizes,
                color=classification_colors.get(
                    classification,
                    "#7f8c8d",
                ),
                label=classification,
                alpha=0.75,
                edgecolors="black",
            )

            for _, row in group.iterrows():
                axes[1, 0].annotate(
                    row["product_name"],
                    (
                        row["units_sold"],
                        row["cm_percentage"],
                    ),
                    xytext=(5, 6),
                    textcoords="offset points",
                    fontsize=8,
                )

        median_volume = df["units_sold"].median()
        median_margin = df["cm_percentage"].median()

        axes[1, 0].axvline(
            median_volume,
            color="gray",
            linestyle="--",
            label=(
                f"Median Volume: {median_volume:.0f}"
            ),
        )

        axes[1, 0].axhline(
            median_margin,
            color="black",
            linestyle="--",
            label=(
                f"Median Margin: {median_margin:.1f}%"
            ),
        )

        axes[1, 0].set_title(
            "Product Volume and Margin Matrix"
        )

        axes[1, 0].set_xlabel("Units Sold")

        axes[1, 0].set_ylabel(
            "Contribution Margin (%)"
        )

        axes[1, 0].grid(
            linestyle="--",
            alpha=0.4,
        )

        axes[1, 0].legend(fontsize=8)

        # الرسم الرابع: توزيع التصنيفات
        classification_counts = (
            df["classification"].value_counts()
        )

        pie_colors = [
            classification_colors.get(
                classification,
                "#7f8c8d",
            )
            for classification
            in classification_counts.index
        ]

        axes[1, 1].pie(
            classification_counts.values,
            labels=classification_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=pie_colors,
        )

        axes[1, 1].set_title(
            "Product Classification Distribution"
        )

        plt.tight_layout(
            rect=(0, 0, 1, 0.96)
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(figure)

        print(
            f"Product charts saved: {save_path}"
        )

    # ==================================================
    # Monthly Charts
    # ==================================================

    def create_monthly_revenue_charts(
        self,
        monthly_analysis: pd.DataFrame,
        save_path: str = "monthly_revenue_analysis.png",
    ) -> None:
        """
        إنشاء لوحة رسوم بيانية للتحليل المالي الشهري.
        """

        if monthly_analysis.empty:
            print(
                "No monthly data available for charts."
            )
            return

        df = monthly_analysis.copy()

        figure, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(16, 11),
        )

        figure.suptitle(
            "Monthly Financial Analysis",
            fontsize=18,
            fontweight="bold",
        )

        month_labels = df["month_name"]
        x_positions = list(range(len(df)))

        # الرسم الأول: الإيرادات الشهرية
        axes[0, 0].plot(
            x_positions,
            df["Revenue"],
            marker="o",
            linewidth=2,
            color="#3498db",
            label="Revenue",
        )

        axes[0, 0].fill_between(
            x_positions,
            df["Revenue"],
            alpha=0.2,
            color="#3498db",
        )

        axes[0, 0].set_title("Monthly Revenue")
        axes[0, 0].set_ylabel("Revenue ($)")
        axes[0, 0].set_xticks(x_positions)

        axes[0, 0].set_xticklabels(
            month_labels,
            rotation=30,
            ha="right",
        )

        axes[0, 0].grid(
            linestyle="--",
            alpha=0.4,
        )

        axes[0, 0].legend()

        for index, revenue in enumerate(
            df["Revenue"]
        ):
            axes[0, 0].annotate(
                f"${revenue:,.0f}",
                (index, revenue),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        # الرسم الثاني: الإيرادات والتكلفة والربح
        bar_width = 0.25

        revenue_positions = [
            position - bar_width
            for position in x_positions
        ]

        profit_positions = [
            position + bar_width
            for position in x_positions
        ]

        axes[0, 1].bar(
            revenue_positions,
            df["Revenue"],
            width=bar_width,
            label="Revenue",
            color="#3498db",
        )

        axes[0, 1].bar(
            x_positions,
            df["COGS"],
            width=bar_width,
            label="COGS",
            color="#e74c3c",
        )

        axes[0, 1].bar(
            profit_positions,
            df["gross_profit"],
            width=bar_width,
            label="Gross Profit",
            color="#2ecc71",
        )

        axes[0, 1].set_title(
            "Revenue, COGS and Gross Profit"
        )

        axes[0, 1].set_ylabel("Amount ($)")
        axes[0, 1].set_xticks(x_positions)

        axes[0, 1].set_xticklabels(
            month_labels,
            rotation=30,
            ha="right",
        )

        axes[0, 1].grid(
            axis="y",
            linestyle="--",
            alpha=0.4,
        )

        axes[0, 1].legend()

        # الرسم الثالث: هوامش الربح
        axes[1, 0].plot(
            x_positions,
            df["gross_margin_pct"],
            marker="o",
            linewidth=2,
            color="#2ecc71",
            label="Gross Margin",
        )

        axes[1, 0].plot(
            x_positions,
            df["operating_margin_pct"],
            marker="s",
            linewidth=2,
            color="#9b59b6",
            label="Operating Margin",
        )

        axes[1, 0].axhline(
            y=0,
            color="black",
            linewidth=1,
        )

        axes[1, 0].set_title(
            "Monthly Profit Margins"
        )

        axes[1, 0].set_ylabel("Margin (%)")
        axes[1, 0].set_xticks(x_positions)

        axes[1, 0].set_xticklabels(
            month_labels,
            rotation=30,
            ha="right",
        )

        axes[1, 0].grid(
            linestyle="--",
            alpha=0.4,
        )

        axes[1, 0].legend()

        # الرسم الرابع: نمو الإيرادات
        growth_values = (
            df["revenue_growth_pct"].fillna(0)
        )

        growth_colors = [
            "#27ae60" if value >= 0 else "#c0392b"
            for value in growth_values
        ]

        axes[1, 1].bar(
            x_positions,
            growth_values,
            color=growth_colors,
        )

        axes[1, 1].axhline(
            y=0,
            color="black",
            linewidth=1,
        )

        axes[1, 1].set_title(
            "Month-over-Month Revenue Growth"
        )

        axes[1, 1].set_ylabel("Growth (%)")
        axes[1, 1].set_xticks(x_positions)

        axes[1, 1].set_xticklabels(
            month_labels,
            rotation=30,
            ha="right",
        )

        axes[1, 1].grid(
            axis="y",
            linestyle="--",
            alpha=0.4,
        )

        for index, value in enumerate(
            growth_values
        ):
            vertical_alignment = (
                "bottom"
                if value >= 0
                else "top"
            )

            axes[1, 1].text(
                index,
                value,
                f"{value:+.1f}%",
                ha="center",
                va=vertical_alignment,
                fontsize=8,
            )

        plt.tight_layout(
            rect=(0, 0, 1, 0.96)
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(figure)

        print(
            f"Monthly charts saved: {save_path}"
        )


# ==================================================
# Profit Optimization Recommender
# ==================================================

class ProfitOptimizationRecommender:
    """
    ينشئ توصيات مالية ويشغل سيناريوهات What-If.
    """

    def __init__(
        self,
        summary: FinancialSummary,
        product_analysis: pd.DataFrame,
    ):
        self.summary = summary
        self.products = product_analysis.copy()

    def generate_strategic_recommendations(
        self,
    ) -> list[dict[str, Any]]:
        """إنشاء توصيات مالية واستراتيجية."""

        recommendations = []

        low_margin_products = self.products[
            self.products["cm_percentage"] < 10
        ]

        if not low_margin_products.empty:
            recommendations.append({
                "strategy": (
                    "Product Rationalization / Repricing"
                ),
                "severity": "CRITICAL",
                "finding": (
                    f"{len(low_margin_products)} "
                    "product(s) have contribution "
                    "margin below 10%."
                ),
                "details": low_margin_products[
                    [
                        "product_name",
                        "cm_percentage",
                        "units_sold",
                    ]
                ].to_dict(orient="records"),
                "action": (
                    "Review prices, negotiate supplier "
                    "costs, or discontinue products "
                    "with weak profitability."
                ),
            })

        if (
            self.summary.total_revenue
            < self.summary.break_even_revenue
        ):
            deficit = (
                self.summary.break_even_revenue
                - self.summary.total_revenue
            )

            recommendations.append({
                "strategy": (
                    "Break-Even Deficit Reduction"
                ),
                "severity": "CRITICAL",
                "finding": (
                    "Revenue is below break-even by "
                    f"${deficit:,.2f}."
                ),
                "details": {
                    "current_revenue":
                        self.summary.total_revenue,
                    "break_even_revenue":
                        self.summary.break_even_revenue,
                },
                "action": (
                    "Reduce unnecessary operating "
                    "expenses and focus on products "
                    "with high margins."
                ),
            })

        elif self.summary.operating_margin_pct < 15:
            recommendations.append({
                "strategy": (
                    "Operating Margin Expansion"
                ),
                "severity": "MEDIUM",
                "finding": (
                    "Operating margin is "
                    f"{self.summary.operating_margin_pct:.2f}%."
                ),
                "details": {
                    "ebit":
                        self.summary.operating_income_ebit,
                },
                "action": (
                    "Review overhead expenses and "
                    "improve operational efficiency."
                ),
            })

        niche_products = self.products[
            self.products["classification"]
            == "Niche Opportunities"
        ]

        if not niche_products.empty:
            recommendations.append({
                "strategy": "Sales Engine Realignment",
                "severity": "GROWTH",
                "finding": (
                    f"{len(niche_products)} high-margin "
                    "product(s) have low sales volume."
                ),
                "details": niche_products[
                    [
                        "product_name",
                        "cm_percentage",
                        "units_sold",
                    ]
                ].to_dict(orient="records"),
                "action": (
                    "Increase advertising and sales "
                    "efforts for these products."
                ),
            })

        if not recommendations:
            recommendations.append({
                "strategy": "Maintain Performance",
                "severity": "LOW",
                "finding": (
                    "No major profitability issues detected."
                ),
                "details": {},
                "action": (
                    "Continue monitoring margins, "
                    "expenses, and product performance."
                ),
            })

        return recommendations

    def run_sensitivity_analysis(
        self,
        price_delta_pct: float = 0.0,
        cogs_delta_pct: float = 0.0,
        volume_delta_pct: float = 0.0,
    ) -> dict[str, float | None]:
        """
        محاكاة أثر تغيير السعر والتكلفة
        وحجم المبيعات على EBIT.
        """

        price_factor = (
            1 + price_delta_pct / 100
        )

        cogs_factor = (
            1 + cogs_delta_pct / 100
        )

        volume_factor = (
            1 + volume_delta_pct / 100
        )

        if price_factor < 0:
            raise ValueError(
                "Price change produces a negative price."
            )

        if cogs_factor < 0:
            raise ValueError(
                "COGS change produces a negative cost."
            )

        if volume_factor < 0:
            raise ValueError(
                "Volume change produces negative volume."
            )

        projected_revenue = (
            self.summary.total_revenue
            * price_factor
            * volume_factor
        )

        projected_cogs = (
            self.summary.cogs
            * cogs_factor
            * volume_factor
        )

        projected_gross_profit = (
            projected_revenue
            - projected_cogs
        )

        projected_ebit = (
            projected_gross_profit
            - self.summary.operating_expenses
        )

        current_ebit = (
            self.summary.operating_income_ebit
        )

        if current_ebit == 0:
            ebit_growth_pct = None
        else:
            ebit_growth_pct = (
                (projected_ebit - current_ebit)
                / abs(current_ebit)
                * 100
            )

        return {
            "projected_revenue":
                round(projected_revenue, 2),
            "projected_cogs":
                round(projected_cogs, 2),
            "projected_gross_profit":
                round(projected_gross_profit, 2),
            "projected_ebit":
                round(projected_ebit, 2),
            "ebit_growth_pct": (
                round(ebit_growth_pct, 2)
                if ebit_growth_pct is not None
                else None
            ),
        }


# ==================================================
# Printing Functions
# ==================================================

def print_financial_summary(
    summary: FinancialSummary,
) -> None:
    """طباعة الملخص المالي."""

    print("\n=== FINANCIAL SUMMARY ===")

    print(
        f"Revenue:              "
        f"${summary.total_revenue:,.2f}"
    )

    print(
        f"COGS:                 "
        f"${summary.cogs:,.2f}"
    )

    print(
        f"Gross Profit:         "
        f"${summary.gross_profit:,.2f} "
        f"({summary.gross_margin_pct:.2f}%)"
    )

    print(
        f"Operating Expenses:   "
        f"${summary.operating_expenses:,.2f}"
    )

    print(
        f"Operating EBIT:       "
        f"${summary.operating_income_ebit:,.2f} "
        f"({summary.operating_margin_pct:.2f}%)"
    )

    if summary.break_even_revenue == float("inf"):
        print(
            "Break-Even Revenue:   "
            "Cannot be calculated"
        )
    else:
        print(
            f"Break-Even Revenue:   "
            f"${summary.break_even_revenue:,.2f}"
        )


def print_monthly_analysis(
    monthly_analysis: pd.DataFrame,
) -> None:
    """طباعة نتائج التحليل المالي الشهري."""

    print("\n=== MONTHLY REVENUE ANALYSIS ===")

    display_columns = [
        "month_name",
        "Revenue",
        "COGS",
        "gross_profit",
        "operating_expenses",
        "operating_income_ebit",
        "gross_margin_pct",
        "operating_margin_pct",
        "revenue_growth_pct",
    ]

    print(
        monthly_analysis[
            display_columns
        ].to_string(
            index=False,
            na_rep="N/A",
        )
    )


def print_product_analysis(
    product_analysis: pd.DataFrame,
) -> None:
    """طباعة نتائج تحليل المنتجات."""

    print("\n=== PRODUCT MATRIX ===")

    print(
        product_analysis[
            [
                "product_name",
                "total_revenue",
                "contribution_margin",
                "cm_percentage",
                "classification",
            ]
        ].to_string(index=False)
    )


# ==================================================
# Main Program
# ==================================================

def main() -> None:

    # بيانات مالية تجريبية لعدة أشهر
    transactions = pd.DataFrame([
        # يناير
        {
            "date": "2026-01-15",
            "category": "Revenue",
            "amount": 150000,
        },
        {
            "date": "2026-01-20",
            "category": "Revenue",
            "amount": 220000,
        },
        {
            "date": "2026-01-22",
            "category": "COGS",
            "amount": 148000,
        },
        {
            "date": "2026-01-25",
            "category": "OpEx_Payroll",
            "amount": 95000,
        },
        {
            "date": "2026-01-28",
            "category": "OpEx_Marketing",
            "amount": 30000,
        },
        {
            "date": "2026-01-30",
            "category": "OpEx_General",
            "amount": 15000,
        },

        # فبراير
        {
            "date": "2026-02-10",
            "category": "Revenue",
            "amount": 240000,
        },
        {
            "date": "2026-02-20",
            "category": "Revenue",
            "amount": 180000,
        },
        {
            "date": "2026-02-22",
            "category": "COGS",
            "amount": 160000,
        },
        {
            "date": "2026-02-25",
            "category": "OpEx_Payroll",
            "amount": 97000,
        },
        {
            "date": "2026-02-26",
            "category": "OpEx_Marketing",
            "amount": 35000,
        },
        {
            "date": "2026-02-28",
            "category": "OpEx_General",
            "amount": 16000,
        },

        # مارس
        {
            "date": "2026-03-12",
            "category": "Revenue",
            "amount": 260000,
        },
        {
            "date": "2026-03-21",
            "category": "Revenue",
            "amount": 230000,
        },
        {
            "date": "2026-03-22",
            "category": "COGS",
            "amount": 185000,
        },
        {
            "date": "2026-03-25",
            "category": "OpEx_Payroll",
            "amount": 100000,
        },
        {
            "date": "2026-03-27",
            "category": "OpEx_Marketing",
            "amount": 38000,
        },
        {
            "date": "2026-03-30",
            "category": "OpEx_General",
            "amount": 17000,
        },

        # أبريل
        {
            "date": "2026-04-11",
            "category": "Revenue",
            "amount": 225000,
        },
        {
            "date": "2026-04-19",
            "category": "Revenue",
            "amount": 205000,
        },
        {
            "date": "2026-04-22",
            "category": "COGS",
            "amount": 172000,
        },
        {
            "date": "2026-04-25",
            "category": "OpEx_Payroll",
            "amount": 101000,
        },
        {
            "date": "2026-04-27",
            "category": "OpEx_Marketing",
            "amount": 32000,
        },
        {
            "date": "2026-04-29",
            "category": "OpEx_General",
            "amount": 17500,
        },
    ])

    # بيانات المنتجات
    products = pd.DataFrame([
        {
            "product_name": "Enterprise Suite",
            "unit_price": 1200,
            "unit_variable_cost": 240,
            "units_sold": 180,
        },
        {
            "product_name": "Mid-Market Core",
            "unit_price": 500,
            "unit_variable_cost": 220,
            "units_sold": 210,
        },
        {
            "product_name": "Starter Kit",
            "unit_price": 80,
            "unit_variable_cost": 75,
            "units_sold": 620,
        },
        {
            "product_name": "Specialized Add-on",
            "unit_price": 400,
            "unit_variable_cost": 80,
            "units_sold": 35,
        },
    ])

    # إنشاء محرك التحليل
    engine = FinancialIntelligenceEngine(
        transactions,
        products,
    )

    # تنفيذ التحليلات
    summary = (
        engine.generate_income_statement()
    )

    monthly_analysis = (
        engine.analyze_monthly_revenue()
    )

    product_analysis = (
        engine.analyze_product_profitability()
    )

    # إنشاء محرك التوصيات
    optimizer = ProfitOptimizationRecommender(
        summary,
        product_analysis,
    )

    recommendations = (
        optimizer
        .generate_strategic_recommendations()
    )

    # طباعة النتائج
    print_financial_summary(summary)

    print_monthly_analysis(
        monthly_analysis
    )

    print_product_analysis(
        product_analysis
    )

    # طباعة التوصيات
    print(
        "\n=== STRATEGIC RECOMMENDATIONS ==="
    )

    for recommendation in recommendations:
        print(
            f"\n[{recommendation['severity']}] "
            f"{recommendation['strategy']}"
        )

        print(
            f"Diagnostic: "
            f"{recommendation['finding']}"
        )

        print(
            f"Action: "
            f"{recommendation['action']}"
        )

    # تحليل What-If
    scenario = optimizer.run_sensitivity_analysis(
        price_delta_pct=5,
        cogs_delta_pct=-3,
        volume_delta_pct=0,
    )

    print("\n=== WHAT-IF SCENARIO ===")

    print(
        f"Projected Revenue:      "
        f"${scenario['projected_revenue']:,.2f}"
    )

    print(
        f"Projected COGS:         "
        f"${scenario['projected_cogs']:,.2f}"
    )

    print(
        f"Projected Gross Profit: "
        f"${scenario['projected_gross_profit']:,.2f}"
    )

    print(
        f"Projected EBIT:         "
        f"${scenario['projected_ebit']:,.2f}"
    )

    if scenario["ebit_growth_pct"] is None:
        print(
            "EBIT Growth:           "
            "Cannot be calculated"
        )
    else:
        print(
            f"EBIT Growth:           "
            f"{scenario['ebit_growth_pct']:+.2f}%"
        )

    # حفظ الجداول في ملفات CSV
    monthly_analysis.to_csv(
        "monthly_revenue_analysis.csv",
        index=False,
    )

    product_analysis.to_csv(
        "product_profitability_analysis.csv",
        index=False,
    )

    # إنشاء رسوم المنتجات
    engine.create_product_charts(
        product_analysis=product_analysis,
        save_path="product_analysis.png",
    )

    # إنشاء الرسوم الشهرية
    engine.create_monthly_revenue_charts(
        monthly_analysis=monthly_analysis,
        save_path="monthly_revenue_analysis.png",
    )

    print("\n=== FILES CREATED ===")
    print("1. product_analysis.png")
    print("2. monthly_revenue_analysis.png")
    print("3. product_profitability_analysis.csv")
    print("4. monthly_revenue_analysis.csv")


if __name__ == "__main__":
    main()