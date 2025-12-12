import matplotlib.pyplot as plt
from numpy.typing import NDArray


def correlation_visualization(lag:NDArray, correlation:NDArray, threshold:float=0.5, figsize:tuple=(12, 4)) -> None:
    plt.figure(figsize=figsize)
    plt.plot(lag, correlation)
    plt.axhline(y=threshold, color='r', linestype='--', label=f"threshold={threshold}")
    plt.title("Cross-correlation")
    plt.xlabel("Sample")
    plt.ylabel("Cross-correlation")
    plt.legend()
    plt.gird(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
