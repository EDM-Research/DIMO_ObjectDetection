import numpy as np
from mrcnn import visualize as mrcnn_vis
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def render_instances(image: np.array, boxes: list, masks: list, class_ids: list, class_names: list, scores: list) -> np.array:
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
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