import numpy as np

def generate_xor(n=500):
    points_per_quadrant = n // 4
    data = []
    labels = []
    data.append(np.random.randn(points_per_quadrant, 2) * 0.3 + np.array([0, 0]))
    labels.extend([0] * points_per_quadrant)
    data.append(np.random.randn(points_per_quadrant, 2) * 0.3 + np.array([1, 1]))
    labels.extend([0] * points_per_quadrant)
    data.append(np.random.randn(points_per_quadrant, 2) * 0.3 + np.array([0, 1]))
    labels.extend([1] * points_per_quadrant)
    data.append(np.random.randn(points_per_quadrant, 2) * 0.3 + np.array([1, 0]))
    labels.extend([1] * points_per_quadrant)
    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels

def save_data(data, labels, filename="data.txt"):
    with open(filename, "w") as f:
        for (x1, x2), y in zip(data, labels):
            f.write(f"{x1} {x2} {y}\n")

if __name__ == "__main__":
    data, labels = generate_xor(500)
    save_data(data, labels)

