import numpy as np
import sympy as sp

# вычисление квадратурной формулы и получение интеграла
def integrate_with_quadrature_method(func, a, b):
    # 3 узла, вес одинаковый
    x0 = a
    x1 = (a + b) / 2
    x2 = b

    w0 = w1 = w2 = 1 / 3 

    f0 = func(x0)
    f1 = func(x1)
    f2 = func(x2)

    # итоговая квадратурная формула
    integral = (w0 * f0 + w1 * f1 + w2 * f2) * (b - a)

    return integral

# функция для вычисления производной
def numerical_derivative(func, x, h=0.001):
    return (func(x + h) - func(x - h)) / (2 * h)

# чтение данных из файла (функция и пределы интегрирования)
def read_function_and_limits_from_file(file_name):
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
        func_str = lines[0].strip()
        a, b = map(float, lines[1].strip().split())
        
        return func_str, a, b
    except Exception as e:
        raise ValueError(f"Ошибка при чтении данных из файла: {e}")

def main():
    try:
        func_str, a, b = read_function_and_limits_from_file('input.txt')
        x = sp.symbols('x')
        func_expr = sp.sympify(func_str)
        func_lambda = sp.lambdify(x, func_expr, 'numpy')

        # высичление интеграла
        result = integrate_with_quadrature_method(func_lambda, a, b)

        # вычичление производной
        derivative_at_0 = numerical_derivative(func_lambda, 0)

        # округление результатов
        result = round(result, 6)
        derivative_at_0 = round(derivative_at_0, 6)

				# запись результатов
        with open('output.txt', 'w') as f:
            f.write(f'Результат интегрирования на интервале [{a}, {b}]: {result}\n')
            f.write(f'Значение производной функции в точке x=0: {derivative_at_0}\n')

    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
