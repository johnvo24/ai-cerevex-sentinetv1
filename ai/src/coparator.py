import matplotlib.pyplot as plt
import numpy as np

class Comparator():
    def plot_model_comparison(metric_names, model1_values, model2_values, model1_label="Model 1", model2_label="Model 2"):
        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width/2, model1_values, width, label=model1_label, color='skyblue')
        bars2 = ax.bar(x + width/2, model2_values, width, label=model2_label, color='lightgreen')

        ax.set_ylabel('Giá trị')
        ax.set_title('So sánh hai mô hình theo các tiêu chí')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()

        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
