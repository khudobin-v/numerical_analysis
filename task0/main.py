import numpy as np

def read_input(file_name):
    with open(file_name, 'r') as file:
        n = int(file.readline().strip())
        a = []
        b = []
        
        for _ in range(n):
            row = list(map(float, file.readline().strip().split()))
            a.append(row[:-1]) 
            b.append(row[-1]) 

    return np.array(a), np.array(b)

def write_output(file_name, solution):
    with open(file_name, 'w') as file:
        for i, x in enumerate(solution):
            file.write(f"x{i + 1} = {x:.2f}\n") 


def gaussian_method(a, b):
    n = len(b)

    for i in range(n):
        max_row_index = np.argmax(np.abs(a[i:n, i])) + i
        a[[i, max_row_index]] = a[[max_row_index, i]]
        b[i], b[max_row_index] = b[max_row_index], b[i]

        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]
            a[j, i:] -= factor * a[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= a[i][j] * x[j]
        x[i] /= a[i][i]

    return x

def main():
    try:
        a, b = read_input("input.txt")
        solution = gaussian_method(a, b)
        write_output("output.txt", solution)
    except FileNotFoundError:
        print("Файл не был найден!")
    except Exception as e:
        print("Ошибка:", e)

if __name__ == "__main__":
    main()
