import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Configura caminhos para importar suas funções
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)

from benchmarks import functions_registry as reg


os.makedirs("images", exist_ok=True)
os.makedirs("images", exist_ok=True)
print("As imagens serão salvas na pasta: images/")

# Carrega funções e limites
_FUNCTIONS = reg.FUNCTIONS
_BOUNDS = reg.BOUNDS
_RESULTS = reg.RESULTS


class NSGA2Optimizer:
    def __init__(self, f1_name, f2_name, pop_size=100, n_gen=200, mutation_rate=0.1):
        # Verificar compatibilidade das funções
        n_var_f1 = len(_BOUNDS[f1_name])
        n_var_f2 = len(_BOUNDS[f2_name])

        if n_var_f1 != n_var_f2:
            raise ValueError(
                f"Funções têm dimensões diferentes: {f1_name}({n_var_f1}) vs {f2_name}({n_var_f2})"
            )

        self.n_var = n_var_f1
        self.xl = _BOUNDS[f1_name][:, 0]
        self.xu = _BOUNDS[f1_name][:, 1]
        self.f1_name = f1_name
        self.f2_name = f2_name
        self.f1_opt = _RESULTS[f1_name]
        self.f2_opt = _RESULTS[f2_name]
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mutation_rate = mutation_rate

    def evaluate(self, X):
        f1 = np.array([_FUNCTIONS[self.f1_name](x) for x in X])
        f2 = np.array([_FUNCTIONS[self.f2_name](x) for x in X])
        return np.column_stack([f1, f2])

    def initialize_population(self):
        return np.array(
            [np.random.uniform(low=self.xl, high=self.xu) for _ in range(self.pop_size)]
        )

    def dominates(self, a, b):
        """Retorna True se a domina b"""
        # a domina b se for melhor em todos os objetivos
        return np.all(a <= b) and np.any(a < b)

    def non_dominated_sort(self, F):
        """Implementação robusta de ordenação não-dominada"""
        n = F.shape[0]
        # Lista para armazenar as frentes
        fronts = []

        # Dominância: [solutions dominated by i, domination count of i]
        S = [[] for _ in range(n)]  # Soluções dominadas por i
        n_count = [0] * n  # Contagem de quantas soluções dominam i
        rank = [0] * n  # Frente de cada solução

        # Preenche as estruturas de dominância
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.dominates(F[i], F[j]):
                    S[i].append(j)
                elif self.dominates(F[j], F[i]):
                    n_count[i] += 1

        # Primeira frente: soluções não-dominadas
        current_front = [i for i in range(n) if n_count[i] == 0]
        fronts.append(current_front)

        # Atualiza o rank das soluções na primeira frente
        for i in current_front:
            rank[i] = 0

        # Constrói frentes subsequentes
        k = 0
        while fronts[k]:
            Q = []  # Próxima frente
            for i in fronts[k]:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        rank[j] = k + 1
                        Q.append(j)
            k += 1
            if Q:
                fronts.append(Q)
            else:
                break

        return fronts

    def crowding_distance(self, F, front):
        """Cálculo robusto de crowding distance"""
        n = len(front)
        distances = np.zeros(n)
        if n <= 2:
            # Se houver 2 ou menos soluções, todas têm distância infinita
            return np.full(n, np.inf)

        # Para cada objetivo
        for m in range(F.shape[1]):
            # Ordena a frente pelo objetivo m
            sorted_front = sorted(front, key=lambda i: F[i, m])
            f_min = F[sorted_front[0], m]
            f_max = F[sorted_front[-1], m]
            scale = f_max - f_min

            # Se todos os valores forem iguais, skip
            if scale < 1e-12:
                continue

            # Define os extremos com distância infinita
            distances[0] = np.inf
            distances[-1] = np.inf

            # Calcula distâncias para os pontos intermediários
            for i in range(1, n - 1):
                prev_idx = sorted_front[i - 1]
                next_idx = sorted_front[i + 1]

                distances[i] += (F[next_idx, m] - F[prev_idx, m]) / scale

        return distances

    def selection(self, population, F):
        """Seleção com tratamento de casos especiais"""
        fronts = self.non_dominated_sort(F)
        new_population = []
        new_F = []
        current_size = 0

        for front in fronts:
            if current_size >= self.pop_size:
                break

            # Calcula crowding distance para a frente atual
            distances = self.crowding_distance(F, front)

            # Se estamos adicionando toda a frente
            if current_size + len(front) <= self.pop_size:
                indices = front
            else:
                # Ordena por crowding distance (maior primeiro)
                indices = sorted(
                    range(len(front)), key=lambda i: distances[i], reverse=True
                )
                indices = [front[i] for i in indices[: self.pop_size - current_size]]

            new_population.extend(population[indices])
            new_F.extend(F[indices])
            current_size += len(indices)

        return np.array(new_population), np.array(new_F)

    def crossover(self, parent1, parent2):
        """Crossover SBX robusto"""
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        eta_c = 20  # Parâmetro de distribuição

        for i in range(self.n_var):
            if np.random.rand() < 0.8:  # Probabilidade de crossover
                x1 = min(parent1[i], parent2[i])
                x2 = max(parent1[i], parent2[i])
                xl = self.xl[i]
                xu = self.xu[i]

                # Calcula beta
                rand = np.random.rand()
                if rand <= 0.5:
                    beta = (2 * rand) ** (1 / (eta_c + 1))
                else:
                    beta = (1 / (2 - 2 * rand)) ** (1 / (eta_c + 1))

                # Cria filhos
                c1 = 0.5 * ((x1 + x2) - beta * (x2 - x1))
                c2 = 0.5 * ((x1 + x2) + beta * (x2 - x1))

                # Garante que estão dentro dos limites
                child1[i] = np.clip(c1, xl, xu)
                child2[i] = np.clip(c2, xl, xu)

        return child1, child2

    def mutate(self, individual):
        """Mutação polinomial robusta"""
        mutated = np.copy(individual)
        eta_m = 20  # Parâmetro de distribuição

        for i in range(self.n_var):
            if np.random.rand() < self.mutation_rate:
                y = mutated[i]
                yl, yu = self.xl[i], self.xu[i]

                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)

                # Calcula delta_q
                rand = np.random.rand()
                if rand < 0.5:
                    delta_q = (
                        2 * rand + (1 - 2 * rand) * (1 - delta1) ** (eta_m + 1)
                    ) ** (1 / (eta_m + 1)) - 1
                else:
                    delta_q = 1 - (
                        2 * (1 - rand) + 2 * (rand - 0.5) * (1 - delta2) ** (eta_m + 1)
                    ) ** (1 / (eta_m + 1))

                y_new = y + delta_q * (yu - yl)
                mutated[i] = np.clip(y_new, yl, yu)

        return mutated

    def optimize(self):
        """Algoritmo NSGA-II principal com tratamento de erros"""
        try:
            # Inicializa população
            population = self.initialize_population()
            F = self.evaluate(population)

            history = []

            for gen in range(self.n_gen):
                # Seleção
                selected_pop, selected_F = self.selection(population, F)

                # Crossover e mutação
                offspring = []
                for i in range(0, self.pop_size, 2):
                    if i + 1 < self.pop_size:
                        idx1, idx2 = np.random.choice(
                            len(selected_pop), 2, replace=False
                        )
                        parent1 = selected_pop[idx1]
                        parent2 = selected_pop[idx2]

                        child1, child2 = self.crossover(parent1, parent2)
                        child1 = self.mutate(child1)
                        child2 = self.mutate(child2)

                        offspring.append(child1)
                        offspring.append(child2)

                # Avalia a nova geração
                offspring = np.array(offspring)
                F_offspring = self.evaluate(offspring)

                # Combina pais e filhos
                combined_pop = np.vstack([population, offspring])
                combined_F = np.vstack([F, F_offspring])

                # Seleciona a próxima geração
                population, F = self.selection(combined_pop, combined_F)

                # Registra histórico
                min_f1 = np.min(F[:, 0])
                min_f2 = np.min(F[:, 1])
                history.append(
                    {"generation": gen, "best_f1": min_f1, "best_f2": min_f2}
                )

                # Progresso
                if gen % 10 == 0 or gen == self.n_gen - 1:
                    print(
                        f"Gen {gen+1}/{self.n_gen} | {self.f1_name}: {min_f1:.6f} | {self.f2_name}: {min_f2:.6f}"
                    )

            # Frente de Pareto final
            fronts = self.non_dominated_sort(F)
            if fronts:
                pareto_front = F[fronts[0]]
                pareto_solutions = population[fronts[0]]
            else:
                print("Atenção: Nenhuma frente de Pareto encontrada!")
                pareto_front = F
                pareto_solutions = population

            return {
                "pareto_front": pareto_front,
                "pareto_solutions": pareto_solutions,
                "history": history,
                "final_population": population,
                "final_F": F,
            }

        except Exception as e:
            print(f"Erro durante otimização: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    def plot_results(self, result):
        """Visualização robusta dos resultados"""
        if result is None:
            print("Nenhum resultado para plotar")
            return

        pareto_front = result["pareto_front"]
        history = result["history"]

        plt.figure(figsize=(12, 5))

        # Plot 1: Frente de Pareto
        plt.subplot(1, 2, 1)
        plt.scatter(
            pareto_front[:, 0],
            pareto_front[:, 1],
            s=50,
            c="royalblue",
            edgecolor="k",
            alpha=0.8,
            label="Frente de Pareto",
        )

        # Destacar soluções ótimas
        if len(pareto_front) > 0:
            min_f1_idx = np.argmin(pareto_front[:, 0])
            min_f2_idx = np.argmin(pareto_front[:, 1])

            plt.scatter(
                pareto_front[min_f1_idx, 0],
                pareto_front[min_f1_idx, 1],
                s=120,
                c="red",
                marker="*",
                edgecolor="k",
                label=f"Melhor {self.f1_name}",
            )

            plt.scatter(
                pareto_front[min_f2_idx, 0],
                pareto_front[min_f2_idx, 1],
                s=120,
                c="green",
                marker="*",
                edgecolor="k",
                label=f"Melhor {self.f2_name}",
            )

        # Ótimos conhecidos
        plt.axhline(
            y=self.f2_opt,
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"Ótimo global {self.f2_name}",
        )
        plt.axvline(
            x=self.f1_opt,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Ótimo global {self.f1_name}",
        )

        plt.title(f"Frente de Pareto: {self.f1_name} vs {self.f2_name}", fontsize=14)
        plt.xlabel(f"{self.f1_name} (Minimização)", fontsize=12)
        plt.ylabel(f"{self.f2_name} (Minimização)", fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()

        # Plot 2: Evolução dos objetivos
        plt.subplot(1, 2, 2)
        if history:
            generations = [h["generation"] for h in history]
            best_f1 = [h["best_f1"] for h in history]
            best_f2 = [h["best_f2"] for h in history]

            plt.plot(generations, best_f1, "r-", label=f"Melhor {self.f1_name}")
            plt.plot(generations, best_f2, "g-", label=f"Melhor {self.f2_name}")
            plt.title("Evolução dos Objetivos", fontsize=14)
            plt.xlabel("Geração", fontsize=12)
            plt.ylabel("Valor da Função", fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.yscale("log")

        plt.tight_layout()

        # Salvar figura
        fig_name = f"nsga2_{self.f1_name}_{self.f2_name}.png"
        save_path = os.path.join("images", fig_name)
        plt.savefig(save_path, dpi=300)
        print(f"Figura salva como: {save_path}")
        plt.show()

        # Análise das soluções
        if len(pareto_front) > 0:
            min_f1_idx = np.argmin(pareto_front[:, 0])
            min_f2_idx = np.argmin(pareto_front[:, 1])

            print("\n" + "=" * 50)
            print(
                f"RESULTADOS FINAIS: {self.f1_name.upper()} vs {self.f2_name.upper()}"
            )
            print(f"Tamanho da frente de Pareto: {len(pareto_front)} soluções")
            print(f"\nMelhor solução para {self.f1_name}:")
            print(f"  Variáveis: {np.round(result['pareto_solutions'][min_f1_idx], 4)}")
            print(
                f"  Valores: {self.f1_name} = {pareto_front[min_f1_idx, 0]:.6f}, {self.f2_name} = {pareto_front[min_f1_idx, 1]:.6f}"
            )
            print(f"\nMelhor solução para {self.f2_name}:")
            print(f"  Variáveis: {np.round(result['pareto_solutions'][min_f2_idx], 4)}")
            print(
                f"  Valores: {self.f1_name} = {pareto_front[min_f2_idx, 0]:.6f}, {self.f2_name} = {pareto_front[min_f2_idx, 1]:.6f}"
            )
            print("=" * 50)


# Exemplo de uso
if __name__ == "__main__":
    # Configuração do otimizador
    optimizer = NSGA2Optimizer(
        f1_name="sphere",
        f2_name="rastrigin",
        pop_size=50,  # População menor para teste rápido
        n_gen=50,  # Gerações reduzidas
    )

    # Executa a otimização
    result = optimizer.optimize()

    # Plota os resultados
    if result:
        optimizer.plot_results(result)

    print("\n\nExecutando segundo exemplo: Sphere vs Ackley")
    optimizer2 = NSGA2Optimizer(
        f1_name="sphere", f2_name="ackley", pop_size=50, n_gen=50
    )
    result2 = optimizer2.optimize()
    if result2:
        optimizer2.plot_results(result2)
