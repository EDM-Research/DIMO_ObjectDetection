import os

import numpy as np
from mrcnn import visualize as mrcnn_vis
import mrcnn.model as mrcnn_model
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mrcnn.utils import Dataset
from mrcnn.config import Config
import cv2


def get_colors(n: int) -> list:
    colors = []
    h_values = np.linspace(0, 179, n)
    for h in h_values:
        colors.append(tuple(cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]/255.0))
    return colors


def show_results(results: list, dataset: Dataset, config: Config):
    for result in results:
        image, *_ = mrcnn_model.load_image_gt(dataset, config, result['image_id'])
        mrcnn_vis.display_instances(image, result['rois'], result['masks'], result['class_ids'], dataset.class_names)


def save_results(results: list, dataset: Dataset, config: Config, location: str):
    for i, result in enumerate(results):
        image, *_ = mrcnn_model.load_image_gt(dataset, config, result['image_id'])
        plot = render_instances(image, result['rois'], result['masks'], result['class_ids'], dataset.class_names, result['scores'])
        cv2.imwrite(os.path.join(location, f"{str(i).zfill(4)}.png"), cv2.cvtColor(plot, cv2.COLOR_RGB2BGR))


def render_instances(image: np.array, boxes: list, masks: list, class_ids: list, class_names: list, scores: list, class_colors: list = None) -> np.array:
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

    colors = [class_colors[id] for id in class_ids] if class_colors is not None else None

    mrcnn_vis.display_instances(
        image=image,
        boxes=boxes,
        masks=masks,
        class_ids=class_ids,
        class_names=class_names,
        scores=scores,
        ax=ax,
        colors=colors
    )

    canvas.draw()
    plot = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return plot