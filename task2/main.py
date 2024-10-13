import numpy as np
import matplotlib.pyplot as plt

# Алгоритм:
# 1. Создаем таблицу P для промежуточных вычислений
# 2. Заполняем первый столбец известными значениями функции
# 3. Последовательно вычисляем элементы таблицы по формуле Эйткена
# 4. Возвращаем значение в правом верхнем углу таблицы - это и есть результат


def aitken_interpolation(x, y, x_new):
    n = len(x)
    P = np.zeros((n, n))
    P[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            P[i, j] = ((x_new - x[i+j]) * P[i, j-1] + (x[i] - x_new) * P[i+1, j-1]) / (x[i] - x[i+j])
    
    return P[0, n-1]

def approximate_function(x, y, x_new):
    return np.array([aitken_interpolation(x, y, xi) for xi in x_new])

def read_input_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        x = np.array([float(val) for val in lines[0].split()])
        y = np.array([float(val) for val in lines[1].split()])
        x_min, x_max = map(float, lines[2].split())
    return x, y, x_min, x_max

def write_output_data(filename, x, y, x_new, y_new):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Исходные данные:\n")
        f.write(f"x: {x}\n")
        f.write(f"y: {y}\n\n")
        f.write("Аппроксимированные значения:\n")
        for xi, yi in zip(x_new, y_new):
            f.write(f"x: {xi:.2f}, y: {yi:.2f}\n")

def plot_approximation(x, y, x_new, y_new, x_min, x_max, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro', label='Исходные точки')
    plt.plot(x_new, y_new, 'b-', label='Аппроксимация по Эйткену')
    plt.axvline(x=x_min, color='g', linestyle='-.', label='Границы аппроксимации')
    plt.axvline(x=x_max, color='g', linestyle='-.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Аппроксимация табличной функции методом Эйткена')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    x, y, x_min, x_max = read_input_data('input.txt')

    x_new = np.linspace(x_min, x_max, 100)

    y_new = approximate_function(x, y, x_new)

    write_output_data('output.txt', x, y, x_new, y_new)

    plot_approximation(x, y, x_new, y_new, x_min, x_max, 'approximation.png')

    print("Расчеты выполнены. Результаты сохранены в файле output.txt, график сохранен в файле approximation.png")

if __name__ == "__main__":
    main()
