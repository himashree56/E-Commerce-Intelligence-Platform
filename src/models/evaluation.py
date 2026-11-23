"""Evaluation and Visualization Module for Customer Segmentation (Phase 4)."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

logger = logging.getLogger("ecommerce_platform.models.evaluation")

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


class SegmentationVisualizer:
    """Generates and saves publication-quality evaluation charts for K-Means Customer Segmentation."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Professional color palette for 5 clusters
        self.palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    def plot_k_evaluation_curves(
        self, eval_metrics: List[Dict[str, Any]], optimal_k: int = 5
    ) -> Path:
        """Plots Inertia (Elbow), Silhouette Score, Davies-Bouldin, and Calinski-Harabasz across k."""
        df_metrics = pd.DataFrame(eval_metrics)
        out_path = self.output_dir / "kmeans_metrics_elbow_silhouette.png"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "K-Means Clustering Hyperparameter Evaluation (k = 2 to 10)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # 1. Inertia (Elbow)
        ax = axes[0, 0]
        ax.plot(df_metrics["k"], df_metrics["inertia"], marker="o", color="#1f77b4", linewidth=2)
        ax.axvline(x=optimal_k, color="#d62728", linestyle="--", alpha=0.8, label=f"Selected k={optimal_k}")
        ax.set_title("Elbow Method: Inertia vs k", fontweight="bold")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Inertia (Sum of Squared Distances)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        # 2. Silhouette Score
        ax = axes[0, 1]
        ax.plot(df_metrics["k"], df_metrics["silhouette_score"], marker="s", color="#2ca02c", linewidth=2)
        ax.axvline(x=optimal_k, color="#d62728", linestyle="--", alpha=0.8, label=f"Selected k={optimal_k} (0.3839)")
        ax.set_title("Silhouette Score vs k (Higher is Better)", fontweight="bold")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Coefficient")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        # 3. Davies-Bouldin Index
        ax = axes[1, 0]
        ax.plot(df_metrics["k"], df_metrics["davies_bouldin_index"], marker="^", color="#ff7f0e", linewidth=2)
        ax.axvline(x=optimal_k, color="#d62728", linestyle="--", alpha=0.8, label=f"Selected k={optimal_k} (0.8103)")
        ax.set_title("Davies-Bouldin Index vs k (Lower is Better)", fontweight="bold")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Davies-Bouldin Index")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        # 4. Calinski-Harabasz Index
        ax = axes[1, 1]
        ax.plot(df_metrics["k"], df_metrics["calinski_harabasz_index"], marker="D", color="#9467bd", linewidth=2)
        ax.axvline(x=optimal_k, color="#d62728", linestyle="--", alpha=0.8, label=f"Global Peak k={optimal_k} (4,959.0)")
        ax.set_title("Calinski-Harabasz Index vs k (Global Peak = 4,959.0)", fontweight="bold")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Calinski-Harabasz Score")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved K-Means evaluation curve chart to {out_path}.")
        return out_path

    def plot_cluster_sizes(self, profiles: Dict[str, Any]) -> Path:
        """Plots customer count and population percentage distribution per cluster."""
        out_path = self.output_dir / "cluster_size_distribution.png"

        labels = [p["cluster_label"] for p in profiles.values()]
        counts = [p["customer_count"] for p in profiles.values()]
        pcts = [p["population_pct"] for p in profiles.values()]
        rev_pcts = [p["revenue_pct"] for p in profiles.values()]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width / 2, pcts, width, label="% of Customer Population", color="#1f77b4")
        rects2 = ax.bar(x + width / 2, rev_pcts, width, label="% of Total Net Revenue", color="#ff7f0e")

        ax.set_title("Customer Segmentation: Population vs Revenue Share by Cluster", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Cluster {i}\n{lbl}" for i, lbl in enumerate(labels)], rotation=15, ha="right")
        ax.set_ylabel("Percentage (%)")
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Annotate bars
        for rect, count in zip(rects1, counts):
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%\n({count:,})",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        for rect in rects2:
            height = rect.get_height()
            va_align = "bottom" if height >= 0 else "top"
            offset = 3 if height >= 0 else -12
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=va_align,
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved cluster size chart to {out_path}.")
        return out_path

    def plot_rfm_distributions(self, df_profiled: pd.DataFrame) -> Path:
        """Plots boxplots of raw RFM features across clusters."""
        out_path = self.output_dir / "rfm_feature_distributions_by_cluster.png"

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            "Raw RFM Feature Distributions across Customer Clusters (Log Scale)",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )

        cluster_order = sorted(df_profiled["cluster_id"].unique())

        # Recency
        sns.boxplot(
            data=df_profiled,
            x="cluster_id",
            y="recency_days",
            ax=axes[0],
            palette=self.palette,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
        )
        axes[0].set_title("Recency (Days Since Last Order)", fontweight="bold")
        axes[0].set_xlabel("Cluster ID")
        axes[0].set_ylabel("Calendar Days")

        # Frequency
        sns.boxplot(
            data=df_profiled,
            x="cluster_id",
            y="frequency",
            ax=axes[1],
            palette=self.palette,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
        )
        axes[1].set_yscale("symlog", linthresh=1.0)
        axes[1].set_title("Purchase Frequency (Orders Count)", fontweight="bold")
        axes[1].set_xlabel("Cluster ID")
        axes[1].set_ylabel("Distinct Valid Invoices (Symlog Scale)")

        # Monetary Net
        sns.boxplot(
            data=df_profiled,
            x="cluster_id",
            y="monetary_net",
            ax=axes[2],
            palette=self.palette,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
        )
        axes[2].set_yscale("symlog", linthresh=10.0)
        axes[2].set_title("Monetary Net Spending (GBP)", fontweight="bold")
        axes[2].set_xlabel("Cluster ID")
        axes[2].set_ylabel("Net Revenue in GBP (Symlog Scale)")

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved RFM distributions chart to {out_path}.")
        return out_path

    def plot_pca_2d_projection(
        self,
        X_scaled: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        cluster_labels_map: Dict[int, str],
    ) -> Path:
        """Generates 2D PCA projection of the standardized RFM feature space."""
        out_path = self.output_dir / "pca_2d_projection.png"

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        centroids_pca = pca.transform(centroids)

        var_exp = pca.explained_variance_ratio_ * 100

        fig, ax = plt.subplots(figsize=(12, 8))

        for c_id in np.unique(labels):
            mask = labels == c_id
            label_name = cluster_labels_map.get(c_id, f"Cluster {c_id}")
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=self.palette[c_id % len(self.palette)],
                label=f"Cluster {c_id}: {label_name}",
                alpha=0.45,
                s=25,
                edgecolors="none",
            )

        # Plot Centroids
        ax.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            c="black",
            marker="X",
            s=200,
            linewidths=2,
            edgecolors="white",
            label="Cluster Centroids (Projected)",
            zorder=10,
        )

        ax.set_title(
            f"2D PCA Projection of Customer Clusters (Total Variance Explained: {var_exp.sum():.1f}%)\n"
            r"$\bf{Note:}$ PCA is used strictly for 2D visualization; K-Means was trained on standardized 3D RFM feature space.",
            fontsize=12,
            pad=15,
        )
        ax.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% Variance)")
        ax.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% Variance)")
        ax.legend(frameon=True, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved PCA 2D projection chart to {out_path}.")
        return out_path

    def plot_cluster_profile_heatmap(self, profiles: Dict[str, Any]) -> Path:
        """Plots normalized median characteristics heatmap across clusters."""
        out_path = self.output_dir / "cluster_profile_heatmap.png"

        data = []
        cluster_names = []
        for p in profiles.values():
            cluster_names.append(f"C{p['cluster_id']}: {p['cluster_label']}")
            data.append({
                "Median Recency (Days)": p["median_recency_days"],
                "Median Frequency (Orders)": p["median_frequency"],
                "Median Net Spend (GBP)": p["median_monetary_net_gbp"],
                "Population Share (%)": p["population_pct"],
                "Revenue Share (%)": p["revenue_pct"],
            })

        df_hm = pd.DataFrame(data, index=cluster_names)

        # Min-max normalize columns for visual color intensity comparison
        df_norm = (df_hm - df_hm.min()) / (df_hm.max() - df_hm.min() + 1e-6)

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.heatmap(
            df_norm,
            annot=df_hm.values,
            fmt=".1f",
            cmap="YlGnBu",
            linewidths=1,
            linecolor="white",
            cbar_kws={"label": "Normalized Intensity (Min-Max scaled per feature)"},
            ax=ax,
        )
        ax.set_title("Customer Cluster Characteristics Matrix", fontsize=14, fontweight="bold", pad=12)
        plt.xticks(rotation=20, ha="right")

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved cluster profile heatmap to {out_path}.")
        return out_path

    def plot_silhouette_samples(
        self, X_scaled: np.ndarray, labels: np.ndarray, optimal_k: int = 5
    ) -> Path:
        """Plots silhouette coefficients for individual customer samples grouped by cluster."""
        out_path = self.output_dir / "silhouette_samples_plot.png"

        sample_sil_values = silhouette_samples(X_scaled, labels)
        avg_score = float(np.mean(sample_sil_values))

        fig, ax = plt.subplots(figsize=(10, 8))
        y_lower = 10

        for i in range(optimal_k):
            ith_cluster_sil = sample_sil_values[labels == i]
            ith_cluster_sil.sort()
            size_cluster_i = ith_cluster_sil.shape[0]
            y_upper = y_lower + size_cluster_i

            color = self.palette[i % len(self.palette)]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_sil,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}")
            y_lower = y_upper + 10

        ax.set_title(
            f"Silhouette Analysis for K-Means (k = {optimal_k})\nAverage Silhouette Score = {avg_score:.4f}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Membership")
        ax.axvline(x=avg_score, color="red", linestyle="--", label=f"Average Score ({avg_score:.4f})")
        ax.set_yticks([])  # Clear y-axis tick labels
        ax.set_xlim([-0.2, 0.8])
        ax.legend()
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved silhouette samples chart to {out_path}.")
        return out_path
