import os
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def correlation_visualization(
    lag: NDArray,
    correlation: NDArray,
    threshold: float = 0.5,
    figsize: tuple = (12, 4),
    save_path: str | None = None,
) -> None:
    """
    Plot the cross-correlation and optionally save the figure to disk.

    If `save_path` is provided, the plot is saved before being shown.
    """
    plt.figure(figsize=figsize)
    plt.plot(lag, correlation)
    plt.axhline(y=threshold, color="r", linestyle="--", label=f"threshold={threshold}")
    plt.title("Cross-correlation")
    plt.xlabel("Sample")
    plt.ylabel("Cross-correlation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        # If a directory is given, save as 'correlation.png' inside it
        save_path = str(save_path)
        if os.path.isdir(save_path):
            file_path = os.path.join(save_path, "correlation.png")
        else:
            file_path = save_path
        plt.savefig(file_path, bbox_inches="tight")

    plt.show()
