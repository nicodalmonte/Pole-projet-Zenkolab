import matplotlib.pyplot as plt

def plot_data(x, y):
    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Sample Plot')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]
    plot_data(x, y)
    