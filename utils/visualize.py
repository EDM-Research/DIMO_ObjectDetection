import numpy as np
from mrcnn import visualize as mrcnn_vis
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def render_instances(image: np.array, boxes: list, masks: list, class_ids: list, class_names: list, scores: list) -> np.array:
    fig = Figure()
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    ax.margins(0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.axis('off')

    mrcnn_vis.display_instances(
        image=image,
        boxes=boxes,
        masks=masks,
        class_ids=class_ids,
        class_names=class_names,
        scores=scores,
        ax=ax
    )

    canvas.draw()
    plot = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return plot