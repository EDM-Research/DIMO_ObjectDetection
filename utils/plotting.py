import numpy as np
from matplotlib import pyplot as plt

plt.style.use("ggplot")


def plot_epochs(models: list, model_to_title: dict = None):
    import file_io

    with plt.style.context('Solarize_Light2'):
        for model_id in models:
            epochs, aps = file_io.read_model_epochs(model_id)
            plt.plot(epochs, aps, label=model_id if model_to_title is None else model_to_title[model_id])

        plt.legend()
        plt.show()
        plt.xlabel("Epoch")
        plt.ylabel("AP")


def plot_aps():
    labels = ["Variable Size", "Equal Size"]

    real_ap = [70.37, 70.37]
    sim_ap = [17.21, 17.21]
    sim_rand_light = [57.68, 41.02]
    sim_rand_pose = [23.77, 5.22]
    sim_rand_add = [61.09, 40.46]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
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
    ax.legend()
    ax.grid(axis='x')

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_aps()