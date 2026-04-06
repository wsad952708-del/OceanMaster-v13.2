"""
EA/GA — 演化/遺傳演算法最佳化模組

============================================================
🎯 功能說明：
    使用遺傳演算法 (Genetic Algorithm) 進行：
    1. 超參數自動搜索 (Hyperparameter Optimization)
    2. 船隊多船航線全局最佳化 (Fleet Route Optimization)
    3. 漁場選擇組合最佳化 (Fishing Spot Portfolio)

    原理：模擬自然界的「物競天擇、適者生存」
    - 產生一群「候選解」（染色體）
    - 評估每個解的「適應度」（Fitness）
    - 好的解「繁殖」（交叉 Crossover）
    - 偶爾「突變」（Mutation）產生新想法
    - 經過數百代演化，最終存活的就是最佳解

📌 架構狀態：✅ 完整架構  |  ❌ 尚未訓練
📌 缺少什麼：真實漁場熱點數據 + 燃油價格 + 船隊資訊
📌 買家需要：
    1. AI 模型輸出的熱點座標列表
    2. 船隊的船隻數量、出發港口座標
    3. 燃油成本函數
    4. 呼叫 optimize_fleet_routes() 即可得到全局最佳航線
============================================================
"""

import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """
    個體（染色體）— 代表一個「候選解」

    在船隊最佳化的場景中，一個個體代表「所有船的航線分配方案」。
    例如：5 艘船分別去哪 5 個熱點、走什麼路線。
    """
    genes: np.ndarray           # 基因序列（編碼方式依問題而定）
    fitness: float = 0.0        # 適應度分數（越高越好）
    metadata: Dict = field(default_factory=dict)  # 附加資訊


class GeneticOptimizer:
    """
    遺傳演算法最佳化器

    ⚠️ 架構狀態：骨架已完成，尚未與真實數據運行
    ⚠️ 買家使用流程：
        1. 定義適應度函數 (fitness_function)
        2. 設定基因長度與值域
        3. 呼叫 evolve() 開始演化
        4. 取得最佳個體 best_individual
    """

    def __init__(
        self,
        population_size: int = 100,
        gene_length: int = 10,
        gene_range: Tuple[float, float] = (0.0, 1.0),
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_ratio: float = 0.1,
        tournament_size: int = 5
    ):
        """
        Args:
            population_size: 族群大小（候選解的數量）
            gene_length: 基因長度（決策變數的維度）
            gene_range: 基因值域（每個基因的最小值和最大值）
            crossover_rate: 交叉率（兩個父母產生後代的機率）
            mutation_rate: 突變率（基因隨機變異的機率）
            elite_ratio: 菁英比例（直接保留到下一代的頂尖個體比例）
            tournament_size: 錦標賽選擇的競爭人數
        """
        self.population_size = population_size
        self.gene_length = gene_length
        self.gene_range = gene_range
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = max(1, int(population_size * elite_ratio))
        self.tournament_size = tournament_size

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[float] = []

        logger.info(
            f"GA 最佳化器初始化 | pop={population_size}, "
            f"genes={gene_length}, cx={crossover_rate}, mut={mutation_rate}"
        )

    # ============================================================
    # 族群初始化
    # ============================================================
    def _initialize_population(self):
        """隨機生成初始族群"""
        self.population = []
        for _ in range(self.population_size):
            genes = np.random.uniform(
                self.gene_range[0], self.gene_range[1],
                size=self.gene_length
            )
            self.population.append(Individual(genes=genes))

    # ============================================================
    # 選擇 — 錦標賽選擇法
    # ============================================================
    def _tournament_select(self) -> Individual:
        """
        錦標賽選擇

        從族群中隨機抽取 tournament_size 個個體，
        選出其中適應度最高的作為父母。
        """
        candidates = random.sample(self.population, self.tournament_size)
        return max(candidates, key=lambda ind: ind.fitness)

    # ============================================================
    # 交叉 — 模擬混合交配 (BLX-α Crossover)
    # ============================================================
    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        BLX-α 交叉

        兩個父母的基因混合，產生兩個後代。
        在漁業場景中：兩種不同的航線方案「雜交」出新方案。
        """
        if random.random() > self.crossover_rate:
            return (
                Individual(genes=parent1.genes.copy()),
                Individual(genes=parent2.genes.copy())
            )

        alpha = 0.5
        child1_genes = np.zeros(self.gene_length)
        child2_genes = np.zeros(self.gene_length)

        for i in range(self.gene_length):
            low = min(parent1.genes[i], parent2.genes[i])
            high = max(parent1.genes[i], parent2.genes[i])
            span = high - low
            child1_genes[i] = np.random.uniform(
                low - alpha * span, high + alpha * span
            )
            child2_genes[i] = np.random.uniform(
                low - alpha * span, high + alpha * span
            )

        # 限制在值域內
        child1_genes = np.clip(child1_genes, *self.gene_range)
        child2_genes = np.clip(child2_genes, *self.gene_range)

        return (
            Individual(genes=child1_genes),
            Individual(genes=child2_genes)
        )

    # ============================================================
    # 突變 — 高斯突變
    # ============================================================
    def _mutate(self, individual: Individual):
        """
        高斯突變

        以小機率對基因進行微小擾動，
        避免演化卡在局部最優解。
        """
        for i in range(self.gene_length):
            if random.random() < self.mutation_rate:
                sigma = (self.gene_range[1] - self.gene_range[0]) * 0.1
                individual.genes[i] += np.random.normal(0, sigma)
                individual.genes[i] = np.clip(
                    individual.genes[i], *self.gene_range
                )

    # ============================================================
    # 演化主迴圈
    # ============================================================
    def evolve(
        self,
        fitness_function: Callable[[np.ndarray], float],
        num_generations: int = 200,
        verbose: bool = True
    ) -> Individual:
        """
        執行遺傳演算法演化

        Args:
            fitness_function: 適應度函數
                輸入：基因陣列 np.ndarray, shape: (gene_length,)
                輸出：適應度分數 float（越高越好）
            num_generations: 演化代數
            verbose: 是否印出進度

        Returns:
            最佳個體 (最高適應度)

        ⚠️ TODO: 買家需要定義自己的 fitness_function
        ⚠️ 範例（超參數搜索）：
            def fitness(genes):
                lr, batch_size, dropout = genes[0], int(genes[1]), genes[2]
                model = train_model(lr=lr, batch_size=batch_size, dropout=dropout)
                return model.validation_accuracy

            optimizer = GeneticOptimizer(gene_length=3)
            best = optimizer.evolve(fitness_function=fitness)

        ⚠️ 範例（船隊航線最佳化）：
            def fitness(genes):
                # genes 編碼了 5 艘船分別去哪 5 個熱點
                assignments = decode_fleet_assignment(genes)
                total_catch = estimate_total_catch(assignments)
                total_fuel = estimate_total_fuel(assignments)
                return total_catch - 0.3 * total_fuel

            optimizer = GeneticOptimizer(gene_length=5*2)  # 5艘船 × (lat, lon)
            best = optimizer.evolve(fitness_function=fitness)
        """
        self._initialize_population()

        for gen in range(num_generations):
            # 評估適應度
            for ind in self.population:
                ind.fitness = fitness_function(ind.genes)

            # 排序
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            # 記錄最佳
            if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = Individual(
                    genes=self.population[0].genes.copy(),
                    fitness=self.population[0].fitness
                )

            self.history.append(self.best_individual.fitness)

            if verbose and gen % 20 == 0:
                logger.info(
                    f"Generation {gen}/{num_generations} | "
                    f"Best Fitness: {self.best_individual.fitness:.6f}"
                )

            # 菁英保留
            new_population = [
                Individual(genes=ind.genes.copy(), fitness=ind.fitness)
                for ind in self.population[:self.elite_count]
            ]

            # 繁殖下一代
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child1, child2 = self._crossover(parent1, parent2)
                self._mutate(child1)
                self._mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population[:self.population_size]

        logger.info(f"演化完成 | 最佳適應度: {self.best_individual.fitness:.6f}")
        return self.best_individual


# ============================================================
# 預定義的漁業適應度函數骨架
# ============================================================

class FleetFitnessEvaluator:
    """
    船隊適應度評估器

    將遺傳演算法的抽象基因，轉換成具體的漁業決策評估。

    ⚠️ TODO: 買家需要提供：
        1. AI 模型輸出的熱點列表 hotspots: [(lat, lon, probability), ...]
        2. 船隊資訊 fleet: [{'name': '...', 'port_lat': ..., 'port_lon': ..., 'fuel_rate': ...}]
        3. 燃油單價 fuel_cost_per_km: float
    """

    def __init__(
        self,
        hotspots: Optional[List[Tuple[float, float, float]]] = None,
        fleet: Optional[List[Dict]] = None,
        fuel_cost_per_km: float = 50.0
    ):
        """
        Args:
            hotspots: 熱點列表 [(lat, lon, catch_probability), ...]
            fleet: 船隻列表 [{'name': str, 'port_lat': float, 'port_lon': float, ...}]
            fuel_cost_per_km: 每公里燃油成本 (TWD)
        """
        self.hotspots = hotspots or []
        self.fleet = fleet or []
        self.fuel_cost_per_km = fuel_cost_per_km

    def evaluate(self, genes: np.ndarray) -> float:
        """
        評估一組船隊分配方案的適應度

        ⚠️ TODO: 買家需要接入真實的熱點數據和船隊資訊
        ⚠️ 目前為架構骨架，返回隨機值

        Args:
            genes: 編碼的船隊分配方案

        Returns:
            適應度分數（預期收益 - 燃油成本）
        """
        raise NotImplementedError(
            "⚠️ 船隊適應度評估需要真實數據。\n"
            "買家請提供：\n"
            "  1. AI 模型輸出的漁場熱點列表\n"
            "  2. 船隊的船隻數量與出發港口座標\n"
            "  3. 每公里燃油成本\n"
            "然後實作此函數的計算邏輯。"
        )


class HyperparameterSearcher:
    """
    超參數搜索器

    使用遺傳演算法自動搜索 DL 模型的最佳超參數組合。
    比 Grid Search 聰明 10 倍，比 Random Search 聰明 5 倍。

    ⚠️ TODO: 買家需要提供一個 train_and_evaluate 函數
    ⚠️ 範例：
        searcher = HyperparameterSearcher()
        best_params = searcher.search(
            param_spaces={
                'learning_rate': (1e-5, 1e-2),
                'batch_size': (8, 128),
                'dropout': (0.0, 0.5),
                'num_layers': (2, 8),
                'hidden_dim': (64, 512),
            },
            train_function=my_train_and_evaluate
        )
    """

    def __init__(self, population_size: int = 50, num_generations: int = 30):
        self.population_size = population_size
        self.num_generations = num_generations

    def search(
        self,
        param_spaces: Dict[str, Tuple[float, float]],
        train_function: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        搜索最佳超參數

        Args:
            param_spaces: 超參數名稱 → (最小值, 最大值)
            train_function: 訓練函數，輸入超參數字典，返回驗證分數

        Returns:
            最佳超參數字典

        ⚠️ TODO: 買家需要提供 train_function
        """
        param_names = list(param_spaces.keys())
        gene_length = len(param_names)
        ranges = list(param_spaces.values())

        optimizer = GeneticOptimizer(
            population_size=self.population_size,
            gene_length=gene_length,
            gene_range=(0.0, 1.0)  # 內部用 [0,1]，再映射到實際值域
        )

        def fitness(genes: np.ndarray) -> float:
            # 將 [0,1] 基因映射到實際值域
            params = {}
            for i, name in enumerate(param_names):
                low, high = ranges[i]
                params[name] = low + genes[i] * (high - low)

            if train_function is None:
                raise NotImplementedError(
                    "⚠️ 需要提供 train_function 來評估超參數。\n"
                    "此函數輸入超參數字典，返回驗證集分數。"
                )

            return train_function(params)

        best = optimizer.evolve(fitness, num_generations=self.num_generations)

        # 解碼最佳基因
        best_params = {}
        for i, name in enumerate(param_names):
            low, high = ranges[i]
            best_params[name] = low + best.genes[i] * (high - low)

        return best_params
