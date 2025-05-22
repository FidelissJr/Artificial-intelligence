import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import concurrent.futures
import os  # para criar pasta

def ackley(x):
    dim = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / dim)) - np.exp(sum_cos / dim) + 20 + np.e


def griewank(x):
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return sum_part - prod_part + 1


class PSOk:
    def __init__(self, func, n_particles, n_iterations, dim,
                 c1=2.1, c2=2.1, bounds=(-32.768, 32.768), vmax_ratio=0.15):
        self.func = func
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.dim = dim
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds

        self.vmax = (bounds[1] - bounds[0]) * vmax_ratio

        phi = c1 + c2
        if phi <= 4:
            raise ValueError("A soma c1 + c2 deve ser maior que 4 para fator constrição válido.")
        self.chi = 2 / abs(2 - phi - np.sqrt(phi**2 - 4*phi))

        self.position = np.random.uniform(bounds[0], bounds[1], (n_particles, dim))
        self.velocity = np.random.uniform(-self.vmax, self.vmax, (n_particles, dim))

        self.pbest = np.copy(self.position)
        self.pbest_score = np.full(n_particles, np.inf)
        self.gbest = None
        self.gbest_score = np.inf
        self.history = []

    def optimize(self, tol=1e-10):
        for iteration in range(self.n_iterations):
            fitness_vals = np.apply_along_axis(self.func, 1, self.position)

            better_mask = fitness_vals < self.pbest_score
            self.pbest_score[better_mask] = fitness_vals[better_mask]
            self.pbest[better_mask] = self.position[better_mask]

            min_idx = np.argmin(self.pbest_score)
            if self.pbest_score[min_idx] < self.gbest_score:
                self.gbest_score = self.pbest_score[min_idx]
                self.gbest = self.pbest[min_idx].copy()

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            cognitive = self.c1 * r1 * (self.pbest - self.position)
            social = self.c2 * r2 * (self.gbest - self.position)

            self.velocity = self.chi * (self.velocity + cognitive + social)

            self.velocity = np.clip(self.velocity, -self.vmax, self.vmax)
            self.position += self.velocity
            self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

            self.history.append(self.gbest_score)

        return self.gbest, self.gbest_score


def run_psok_run(run_id):
    n_particles = 30
    n_iterations = 1000
    dim = 10
    print(f"Iniciando run {run_id}...") 
    psok = PSOk(ackley, n_particles, n_iterations, dim)
    best_position, best_score = psok.optimize()
    print(f"Run {run_id} concluída: melhor posição {best_position}, melhor score {best_score}")
    return psok.history

# Pasta para salvar os gráficos
output_folder = "pso_convergence_plots"
os.makedirs(output_folder, exist_ok=True)

n_runs = 10

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(run_psok_run, range(1, n_runs + 1)))

# extrair os melhores scores finais de cada run
final_scores = [convergence[-1] for convergence in results]

# calcular média e desvio padrão
mean_score = np.mean(final_scores)
std_score = np.std(final_scores)

print(f"\nMédia dos melhores scores finais: {mean_score}")
print(f"Desvio padrão dos melhores scores finais: {std_score}")

print("\nResultados de todas as runs:")
for i, convergence in enumerate(results, 1):
    print(f"Run {i} Convergência: {convergence[-1]}")

    # Gerar gráfico individual para a run i
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(convergence)), convergence, label=f'Execução {i}')
    plt.xlabel('Iterações')
    plt.ylabel('Melhor Score Global')
    plt.title(f'Gráfico de Convergência da Run {i}')
    plt.legend()
    plt.grid(True)

    # Usar escala logarítmica no eixo y
    plt.yscale('log')

    # Formatador para mostrar 10^x no eixo y
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f'$10^{{{int(np.log10(y))}}}$' if y > 0 else '0'
    ))
    
    # Salvar a imagem na pasta
    filename = os.path.join(output_folder, f'pso_convergence_run_{i}.png')
    plt.savefig(filename)
    plt.close()  # fecha a figura para liberar memória

# Após a execução das 10 runs e antes do print final dos gráficos salvos

plt.figure(figsize=(10, 6))
for i, convergence in enumerate(results, 1):
    plt.plot(range(len(convergence)), convergence, label=f'Run {i}')

plt.xlabel('Iterações')
plt.ylabel('Melhor Resultado Global')
plt.title('Convergência das 10 Runs do PSOk (Ackley - 10D)')
plt.yscale('log')
plt.legend()
plt.grid(True)

# Salvar o gráfico consolidado
filename_all = os.path.join(output_folder, 'pso_convergence_all_runs.png')
plt.savefig(filename_all)
plt.show()
plt.close()

print(f"\nGráficos salvos na pasta '{output_folder}'.")