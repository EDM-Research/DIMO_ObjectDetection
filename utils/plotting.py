from typing import List

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import umap

plt.style.use("ggplot")


def experiment_1_epoch():
    import file_io

    models = [
        "dimo20220411T1045",
        "dimo20220530T1039",
        "dimo20220212T1254",
        "dimo20220214T0430",
        "dimo20220217T1204",
        "dimo20220705T1354",
        "dimo20220707T0242",
        "dimo20220708T1446"
    ]

    labels = [
        "Real",
        "Synth",
        "Synth, rand light",
        "Synth, rand pose",
        "Synth, rand all",
        "Synth, rand light",
        "Synth, rand pose",
        "Synth, rand all"
    ]

    colors = [
        "#E24A33",
        "#348ABD",
        "#988ED5",
        "#777777",
        "#FBC15E",
        "#988ED5",
        "#777777",
        "#FBC15E"
    ]

    lines = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "--",
        "--",
        "--"
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    for model_id, label, color, line in zip(models, labels, colors, lines):
        epochs, aps = file_io.read_model_epochs(model_id)
        ax.plot(epochs, aps, line, label=label, color=color)

    ax.set_title('Performance on Real Test Set')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

    plt.show()

    fig.savefig('../plots/exp1_epochs.png', dpi=300)


def experiment_1():
    labels = ["Variable Size", "Equal Size"]

    real_ap = [70.37, 70.37]
    sim_ap = [17.21, 17.21]
    sim_rand_light = [57.68, 41.02]
    sim_rand_pose = [23.77, 5.22]
    sim_rand_add = [61.09, 40.46]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(8,4))
    rects1 = ax.bar(x - 2 * width, real_ap, width, label='Real')
    rects2 = ax.bar(x - width, sim_ap, width, label='Synth')
    rects3 = ax.bar(x, sim_rand_light, width, label='Synth, rand light')
    rects4 = ax.bar(x + width, sim_rand_pose, width, label='Synth, rand pose')
    rects5 = ax.bar(x + 2 * width, sim_rand_add, width, label='Synth, rand all')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AP')
    ax.set_title('Performance on Real Test Set')
    ax.set_xticks(x, labels)
    ax.set_ylim([0, 100])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(axis='x')

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)

    fig.tight_layout()

    plt.show()

    fig.savefig('../plots/exp1.png', dpi=300)


def experiment_2():
    labels = ["Data Augmentation", "Transfer Learning", "Both", "Both (equal size)"]

    real_ap = [74.1, 78.8, 77.9, 77.9]
    sim_ap = [53.2, 77.3, 78.5, 78.5]
    sim_rand_light = [55.3, 76.4, 75.7, 75.4]
    sim_rand_pose = [36.7, 69.3, 67.9, 65.8]
    sim_rand_add = [61.5, 69.6, 71.5, 69.4]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    start_x = 0 - (2 * width + .5 * width) - 0.05
    end_x = x[-1] + 2 * width + .5 * width

    fig, ax = plt.subplots(figsize=(12,4))
    rects1 = ax.bar(x - 2 * width, real_ap, width, label='Real')
    ax.plot([end_x, start_x], [70.37, 70.37], '--', zorder=-1, linewidth=1)
    ax.text(start_x - 0.15, 70.37 - 2, "70.4", c="#E24A33")

    rects2 = ax.bar(x - width, sim_ap, width, label='Synth')
    ax.plot([end_x, start_x], [17.21, 17.21], '--', zorder=-1, linewidth=1)
    ax.text(start_x - 0.15, 17.21 - 2, "17.2", c="#348ABD")

    rects3 = ax.bar(x, sim_rand_light, width, label='Synth, rand light')
    ax.plot([end_x, start_x], [57.68, 57.68], '--', zorder=-1, linewidth=1)
    ax.text(start_x - 0.15, 57.68 - 2, "57.7", c="#988ED5")

    rects4 = ax.bar(x + width, sim_rand_pose, width, label='Synth, rand pose')
    ax.plot([end_x, start_x], [23.77, 23.77], '--', zorder=-1, linewidth=1)
    ax.text(start_x - 0.15, 23.77 - 2, "23.8", c="#777777")

    rects5 = ax.bar(x + 2 * width, sim_rand_add, width, label='Synth, rand all')
    ax.plot([end_x, start_x], [61.09, 61.09], '--', zorder=-1, linewidth=1)
    ax.text(start_x - 0.15, 61.09 - 2, "61.1", c="#FBC15E")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AP')
    ax.set_title('Performance on Real Test Set')
    ax.set_xticks(x, labels)
    ax.set_ylim([0, 100])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(axis='x')

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)

    fig.tight_layout()

    plt.show()

    fig.savefig('../plots/exp2.png', dpi=300)


def experiment_2_epoch():
    import file_io

    models = [
        "dimo20220415T0810", # aug
        "dimo20220417T1814", # aug
        "dimo20220210T0254", # aug
        "dimo20220217T0257", # aug
        "dimo20220222T1437", # aug
        "dimo20220419T1045", # tl
        "dimo20220420T1506", # tl
        "dimo20220422T0919", # tl
        "dimo20220424T2051", # tl
        "dimo20220426T1027" # tl
    ]

    labels = [
        "Real",
        "Synth",
        "Synth, rand light",
        "Synth, rand pose",
        "Synth, rand all",
        "Real",
        "Synth",
        "Synth, rand light",
        "Synth, rand pose",
        "Synth, rand all"
    ]

    colors = [
        "#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E",
        "#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E"
    ]

    lines = [
        "-", "-", "-", "-", "-",
        ":", ":", ":", ":", ":"
    ]

    fig, ax = plt.subplots()

    for model_id, label, color, line in zip(models, labels, colors, lines):
        epochs, aps = file_io.read_model_epochs(model_id)
        ax.plot(epochs, aps, line, label=label, color=color)

    ax.set_title('Performance on Real Test Set')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

    plt.show()

    fig.savefig('../plots/exp2_epochs.png', dpi=300)



def experiment_3():
    matplotlib.rcParams['xtick.minor.size'] = 0
    matplotlib.rcParams['xtick.minor.width'] = 0

    ap_values = [69.42, 72.23, 72.58, 73.01, 72.01, 71.52]
    image_counts = [1755, 4387, 8775, 17550, 35100, 70200]

    fig, ax = plt.subplots()
    ax.plot(image_counts, ap_values, 'o-', c="#FBC15E")

    ax.set_ylabel('AP')
    ax.set_xlabel('Image Count (log scale)')
    ax.set_xscale("log")
    ax.set_xticks(image_counts)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title('Performance on Real Test Set')

    plt.show()

    fig.savefig('../plots/exp3.png', dpi=300)


def experiment_4():
    fig, ax = plt.subplots()

    ap_values = [81.26, 76.71, 80.77, 77.13, 71.52]
    labels = ["All", "3+", "4+", "5+", "Heads"]

    rects = ax.bar(labels, ap_values, color="#FBC15E")
    ax.set_xlabel("Layers Retrained")
    ax.set_ylabel("AP")
    ax.set_ylim([0, 100])

    ax.bar_label(rects, padding=3)

    plt.show()

    fig.savefig('../plots/exp4.png', dpi=300)


def experiment_5():
    labels = ["1:0", "20:1", "10:1", "5:1", "2:1", "1:1"]

    mixed = [73.44, 75.90, 78.00, 80.13, 79.26, 79.02]
    finetune = [73.44, 82.05, 82.36, 83.00, 83.40, 83.70]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    start_x = 0 - (width) - 0.1
    end_x = x[-1] + .5 * width

    fig, ax = plt.subplots(figsize=(10,4))
    rects1 = ax.bar(x - 0.5 * width, mixed, width, label='Mixed', color="#FFB5B8")
    ax.plot([end_x, start_x], [77.91, 77.91], '--', zorder=-1, linewidth=1, c="#E24A33")
    ax.text(start_x - 0.4, 77.91 - 2, "77.91", c="#E24A33")

    rects2 = ax.bar(x + 0.5 * width, finetune, width, label='Finetuning', color="#8EBA42")
    ax.plot([end_x, start_x], [71.52, 71.52], '--', zorder=-1, linewidth=1, c="#FBC15E")
    ax.text(start_x - 0.4, 71.52 - 2, "71.52", c="#FBC15E")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AP')
    ax.set_title('Performance on Real Test Set')
    ax.set_xlabel("Ratio Synthetic:Real")
    ax.set_xticks(x, labels)
    ax.set_ylim([0, 100])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(axis='x')

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

    fig.savefig('../plots/exp5.png', dpi=300)


def plot_feature_maps(embeddings_per_level: List[np.array], titles: list = None):
    titles = list(map(str, range(len(embeddings_per_level[0])))) if titles is None else titles

    fig, ax = plt.subplots(1, len(embeddings_per_level), figsize=(10,4))

    for i, level_embeddings in enumerate(embeddings_per_level):
        for embedding, title in zip(level_embeddings, titles):
            ax[i].scatter(embedding[:,0], embedding[:,1], label=title, s=0.5)
            ax[i].set_title(f"Level {i}")

    ax[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    experiment_2_epoch()