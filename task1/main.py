import numpy as np

def gauss_seidel(a, b, x_init=None, tolerance=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros_like(b, dtype=np.double) if x_init is None else x_init.copy()
    
    for iteration in range(max_iterations):
        x_old = x.copy()
        
        for i in range(n):
            sum_ax = np.dot(a[i], x) - a[i][i] * x[i]  # Вычисляем сумму для текущего уравнения
            x[i] = (b[i] - sum_ax) / a[i][i]  # Обновляем значение x[i]

        # Проверяем сходимость
        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            print(f"Сошлось за {iteration + 1} итерацию(-ий).")
            return x

    print("Максимальное количество итераций достигнуто, решение может быть не найдено.")
    return x

def read_input(file_name):
    with open(file_name, 'r') as file:
        n = int(file.readline().strip())
        a = []
        b = []
        
        for _ in range(n):
            row = list(map(float, file.readline().strip().split()))
            a.append(row[:-1])  # Все элементы, кроме последнего
            b.append(row[-1])   # Последний элемент

    return np.array(a), np.array(b)

def write_output(file_name, solution):
    with open(file_name, 'w') as file:
        for i, x in enumerate(solution):
            file.write(f"x{i + 1} = {x:.4f}\n")  # Ограничиваем до 4 знаков после запятой

def main():
    try:
        a, b = read_input("input.txt")
        solution = gauss_seidel(a, b)
        write_output("output.txt", solution)
    except FileNotFoundError:
        print("Файл не был найден!")
    except Exception as e:
        print("Ошибка:", e)

if __name__ == "__main__":
    main()
