import matplotlib.pyplot as plt
from numpy.typing import NDArray


def correlation_visualization(lag:NDArray, correlation:NDArray, threshold:float=0.5, figsize:tuple=(12, 4)) -> None:
    plt.figure(figsize=figsize)
    plt.plot(lag, correlation)
    # matplotlib uses 'linestyle', not 'linestype'
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"threshold={threshold}")
    plt.title("Cross-correlation")
    plt.xlabel("Sample")
    plt.ylabel("Cross-correlation")
    plt.legend()
    # fix typo: 'grid' instead of 'gird'
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
