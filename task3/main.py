import numpy as np
import sympy as sp

def parse_input(filename):
    """Парсинг данных из файла input.txt"""
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    equation_line = next(line for line in lines if 'f(x) =' in line)
    initial_guess_line = next(line for line in lines if 'x0 =' in line)
    tolerance_line = next(line for line in lines if 'tol =' in line)
    
    # Получаем уравнение, начальное приближение и точность
    equation_str = equation_line.split('=')[1].strip()
    initial_guess = float(initial_guess_line.split('=')[1].strip())
    tolerance = float(tolerance_line.split('=')[1].strip())
    
    return equation_str, initial_guess, tolerance

def aitken_delta_squared(f, x0, tol, q):
    """Решение нелинейного уравнения методом простой итерации с процессом Эйткена"""
    x = sp.symbols('x')
    f_expr = sp.sympify(f)
    
    # Проблема в том, что мы пытаемся найти неподвижную точку f(x),
    # но нам нужно найти корень. Преобразуем уравнение f(x) = 0 в x = g(x)
    g_expr = x + f_expr  # g(x) = x + f(x) - одна из возможных форм
    g_func = sp.lambdify(x, g_expr, 'numpy')
    
    # Начальные значения
    x1 = g_func(x0)
    x2 = g_func(x1)
    
    iteration = 0
    max_iter = 100
    
    while iteration < max_iter:
        try:
            # Этап ускорения Δ^2 по формуле Эйткена
            x_tilde_2 = (x0 * x2 - x1**2) / (x2 - 2 * x1 + x0)
            
            # Вычисление контрольного значения
            x3 = g_func(x_tilde_2)
            
            # Проверка сходимости
            if abs(x3 - x_tilde_2) < tol:
                return x_tilde_2
            
            # Обновление значений для следующей итерации
            x0, x1 = x_tilde_2, x3
            x2 = g_func(x1)
            
        except (ZeroDivisionError, RuntimeWarning):
            print("Ошибка: деление на ноль или числовая нестабильность")
            return None
            
        iteration += 1
    
    print("Превышено максимальное количество итераций")
    return None

def main():
    # Чтение данных из файла
    equation_str, x0, tol = parse_input('input.txt')
    q = 0.5  # этот параметр теперь не используется
    
    # Вычисление корня
    root = aitken_delta_squared(equation_str, x0, tol, q)
    
    if root is not None:
        # Проверка результата
        x = sp.symbols('x')
        f_expr = sp.sympify(equation_str)
        f_func = sp.lambdify(x, f_expr, 'numpy')
        residual = abs(f_func(root))
        
        result_str = f"Приближенное значение корня: {root}\n"
        result_str += f"Невязка |f(x)|: {residual}"
        
        print(result_str)
        
        # Запись результата в файл
        with open('output.txt', 'w') as file:
            file.write(result_str)
    else:
        error_msg = "Не удалось найти корень. Попробуйте другое начальное приближение."
        print(error_msg)
        with open('output.txt', 'w') as file:
            file.write(error_msg)

if __name__ == "__main__":
    main()