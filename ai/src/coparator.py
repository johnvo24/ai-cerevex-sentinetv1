import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

class Comparator():
    
    def plot_avg_metrics(
        avg_accu_rl, avg_pred_time_rl, avg_readed_token_rl,
        avg_accu_sft, avg_pred_time_sft, avg_readed_token_sft,
        save_path="res/evaluation/rl_avg_metrics_comparison.png"
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Accuracy plot
        metric_names_acc = ['SFT+RL', 'SFT']
        values_acc = [avg_accu_rl, avg_accu_sft]
        bars1 = ax1.bar(metric_names_acc, values_acc, color=['skyblue', 'lightgreen'])
        ax1.set_title("Accuracy Comparison")
        ax1.set_ylim(0, 1)
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Prediction time plot
        metric_names_time = ['SFT+RL', 'SFT']
        values_time = [avg_pred_time_rl, avg_pred_time_sft]
        bars2 = ax2.bar(metric_names_time, values_time, color=['skyblue', 'lightgreen'])
        ax2.set_title("Prediction Time Comparison (seconds)")
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Readed tokens plot
        metric_names_tokens = ['SFT+RL', 'SFT']
        values_tokens = [avg_readed_token_rl, avg_readed_token_sft]
        bars3 = ax3.bar(metric_names_tokens, values_tokens, color=['skyblue', 'lightgreen'])
        ax3.set_title("Avg. Readed Tokens Comparison")
        for bar in bars3:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def save_classification_report_as_image(true_labels, pred_labels, target_names=None, save_path="res/evaluation/sft_classification_report.png"):
        report = classification_report(true_labels, pred_labels, target_names=target_names)
        print(report)
        fig = plt.figure(figsize=(10, 6))
        plt.text(0, 1, report, fontsize=12, family='monospace', verticalalignment='top')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)