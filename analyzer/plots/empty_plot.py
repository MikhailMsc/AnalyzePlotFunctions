from matplotlib import pyplot as plt

from .config import PlotConfig


def plot_empty_area(text: str = '', plot_config: PlotConfig = None):
    fig = plt.figure(figsize=plot_config.plot_size)
    ax = fig.add_subplot(111)
    _ = ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=plot_config.annotation_font_size)
    plt.axis('off')  # Скрываем оси
    plt.show()
