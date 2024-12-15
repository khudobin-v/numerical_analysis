import numpy as np
import sympy as sp
import re

def read_equations(filename):
    with open(filename, 'r') as file:
        equations = file.readlines()
    return [equation.strip() for equation in equations]

# фунцкия для парсинга уравений
def parse_equations(equations):
    symbols = set()
    for eq in equations:
        symbols.update(re.findall(r'[a-zA-Z]+', eq))
    symbols = list(symbols)
    sym_vars = sp.symbols(symbols)
    
    functions = []
    
    for eq in equations:

        eq = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', eq) 
        eq = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', eq)  
        
        try:
            sym_eq = sp.sympify(eq)
            f = sp.lambdify(sym_vars, sym_eq, 'numpy')
            functions.append(f)
        except Exception as e:
            print(f"Ошибка при разборе уравнения: {eq}. Ошибка: {e}")
            raise
    
    return functions, sym_vars

# функция для вычисления Якобиана
def jacobian(f, X, sym_vars):
    jac = np.zeros((len(sym_vars), len(sym_vars)))
    h = 1e-6  # малое значение для численного дифференцирования

    for i in range(len(sym_vars)):
        for j in range(len(sym_vars)):
            X1 = X.copy()
            X1[j] += h
            jac[i, j] = (f[i](*X1) - f[i](*X)) / h
    return jac

# метод Ньютона для нахождения решения нелинейной системы
def newton_method(f, X, sym_vars):
    max_iter = 100
    tol = 1e-6
    for _ in range(max_iter):
        F = np.array([eq(*X) for eq in f])
        J = jacobian(f, X, sym_vars)
        delta_X = np.linalg.solve(J, -F)
        X = X + delta_X
        if np.linalg.norm(delta_X) < tol:
            return X
    raise ValueError("Метод Ньютона не сошелся")

# метод Зейделя с внутренним методом Ньютона
def gauss_seidel_method(f, initial_guess, sym_vars, max_outer_iter=50, tol=1e-6):
    X = np.array(initial_guess, dtype=float)
    for _ in range(max_outer_iter):
        X_new = X.copy()
        X_new = newton_method(f, X_new, sym_vars)
        if np.linalg.norm(X_new - X) < tol:
            return X_new
        X = X_new
    raise ValueError("Метод Зейделя не сошелся")

# запись результата в файл
def write_output(filename, solution, sym_vars, precision=6):
    with open(filename, 'w') as file:
        for var, sol in zip(sym_vars, solution):
            file.write(f"{var} = {sol:.{precision}f}\n")

def main():
    equations = read_equations('input.txt')
    functions, sym_vars = parse_equations(equations)
    initial_guess = [0.0] * len(sym_vars)
    solution = gauss_seidel_method(functions, initial_guess, sym_vars)
    write_output('output.txt', solution, sym_vars)

if __name__ == "__main__":
    main()
