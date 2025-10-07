# Ranktration - RULER for Elixir

[![Hex.pm](https://img.shields.io/hexpm/v/ranktration.svg)](https://hex.pm/packages/ranktration)
[![Hex.pm](https://img.shields.io/hexpm/dt/ranktration.svg)](https://hex.pm/packages/ranktration)

**RULER - Robust Unified Learning Evaluation & Ranking**

A domain-independent trajectory ranking framework that can evaluate and rank any quantifiable attempts, methods, or solutions across any field - from algorithms to AI models to optimization techniques.

## Philosophical Foundation

RULER (Ranktraction) changes the question from *"What's the optimal solution?"* to *"Which established approach works best given our measurements?"*

Instead of expensive global optimization search, RULER focuses on **intelligent evaluation** of completed trajectories using multi-criteria analysis, statistical confidence, and consensus ranking.

## Quick Start

### Installation

```elixir
def deps do
  [
    {:ranktration, "~> 0.1.0"}
  ]
end
```

### Basic Usage

```elixir
# Create trajectories with your domain's metrics
trajectories = [
  Ranktration.TrajectoryResult.new("timsort_sort", "sorting_benchmark", %{
    "execution_time" => 0.85,   # Fast
    "space_efficiency" => 0.90, # Efficient
    "correctness" => 1.0,       # Always correct
    "stability" => 1.0          # Preserves order
  }),
  Ranktration.TrajectoryResult.new("heapsort_sort", "sorting_benchmark", %{
    "execution_time" => 0.75,   # Decent speed
    "space_efficiency" => 0.95, # Very efficient
    "correctness" => 1.0,       # Always correct
    "stability" => 0.0          # Unstable sorting
  })
]

# Configure evaluation weights for your use case
evaluator = Ranktration.RulerCore.new(metric_weights: %{
  "execution_time" => 0.4,     # Speed matters most
  "correctness" => 0.3,        # Must be accurate
  "stability" => 0.2,          # Handle edge cases
  "space_efficiency" => 0.1    # Memory secondary
})

# Evaluate and rank
result = Ranktration.RulerCore.evaluate_trajectories(
  evaluator,
  trajectories,
  "sorting_benchmark"
)

# Results
IO.inspect(result.rankings)      # ["timsort_sort", "heapsort_sort"]
IO.inspect(result.scores)        # %{"timsort_sort" => 0.89, "heapsort_sort" => 0.73}
IO.inspect(result.confidence)    # 0.23 (shows meaningful differences)
```

### Domain Examples

#### Algorithm Performance Analysis

```elixir
# Compare sorting algorithms
sorting_algorithms = [
  # Timsort (Python's default) - excellent general purpose
  Ranktration.TrajectoryResult.new("timsort", "sort_bench", %{"speed" => 0.85, "stability" => 1.0, "worst_case" => 0.9}),
  # QuickSort - fast average, bad worst case
  Ranktration.TrajectoryResult.new("quicksort", "sort_bench", %{"speed" => 0.92, "stability" => 0.0, "worst_case" => 0.3}),
]

evaluator = Ranktration.RulerCore.new(metric_weights: %{"speed" => 0.5, "stability" => 0.3, "worst_case" => 0.2})
ranking = Ranktration.RulerCore.evaluate_trajectories(evaluator, sorting_algorithms, "sort_bench")
```

#### Machine Learning Model Comparison

```elixir
# Compare ML models on a dataset
models = [
  Ranktration.TrajectoryResult.new("xgboost", "house_prices", %{"accuracy" => 0.88, "train_time" => 0.7, "interpretability" => 0.6}),
  Ranktration.TrajectoryResult.new("neural_net", "house_prices", %{"accuracy" => 0.91, "train_time" => 0.4, "interpretability" => 0.3}),
]

evaluator = Ranktration.RulerCore.new(metric_weights: %{"accuracy" => 0.5, "train_time" => 0.3, "interpretability" => 0.2})
ranking = Ranktration.RulerCore.evaluate_trajectories(evaluator, models, "house_prices")
```

#### Compiler Optimization Ranking

```elixir
optimizations = [
  Ranktration.TrajectoryResult.new("llvm-o3", "my_app", %{"speed" => 0.95, "code_size" => 0.7, "compile_time" => 0.4}),
  Ranktration.TrajectoryResult.new("clang-oz", "my_app", %{"speed" => 0.85, "code_size" => 0.92, "compile_time" => 0.8}),
]

evaluator = Ranktration.RulerCore.new(metric_weights: %{"speed" => 0.4, "code_size" => 0.4, "compile_time" => 0.2})
ranking = Ranktration.RulerCore.evaluate_trajectories(evaluator, optimizations, "my_app")
```

## How RULER Works

### 1. Trajectory Collection
Gather multiple approaches/solutions with measurable quality metrics.

### 2. Pairwise Analysis
Each trajectory competes against every other trajectory using weighted metrics.

### 3. Consensus Ranking
Algorithm establishes consistent ranking through tournament-style competition.

### 4. Confidence Analysis
Statistical analysis provides confidence intervals and stability measurements.

### 5. Final Scores
Combine metric weights with ranking bonuses for comprehensive evaluation.

## API Reference

### Core Modules

#### `Ranktration.TrajectoryResult`
Represents a single trajectory with quality scores.

- `new/2,3,4`: Create new trajectory
- `has_metric?/2`: Check if metric exists
- `get_metric/2,3`: Retrieve metric value
- `has_all_metrics?/2`: Validate required metrics

#### `Ranktration.RulerCore`
Main evaluation engine with configurable metric weights.

- `new/1`: Initialize with metric weights
- `configure_metrics/2`: Update metric configuration
- `evaluate_trajectories/3`: Complete ranking evaluation

#### `Ranktration.RankingResult`
Contains evaluation results and analysis.

- `top_trajectory/1`: Get highest ranked trajectory
- `bottom_trajectory/1`: Get lowest ranked trajectory
- `get_score/2`: Retrieve final score for trajectory
- `get_rank/2`: Get ranking position (1-indexed)

## Philosophy & Design Principles

### Domain Independence
No assumptions about your problem domain - works with any quantifiable metrics.

### Statistical Rigor
Confidence intervals, consensus analysis, and tie-breaking ensure reliable rankings.

### Scalability
O(n²) pairwise comparisons work excellently for reasonable trajectory counts (dozens to hundreds).

### Extensibility
Easy to add new metrics, weights, or custom comparison functions.

### Reproducibility
Deterministic results with mathematical foundation in tournament theory.

## Performance Characteristics

- **Complexity**: O(n²) for n trajectories (pairwise comparisons)
- **Memory**: O(n²) for storing comparison results
- **Scalability**: Optimal for 2-100 trajectories
- **Speed**: Sub-millisecond evaluation for small datasets

## Compared to Other Approaches

| Approach | Use Case | Strength | Limitation |
|----------|----------|----------|------------|
| **RULER** | Ranking established trajectories | Comprehensive comparison, statistical | Requires pre-existing trajectories |
| **Bayesian Optimization** | Finding optimal configuration | Global optimization search | Expensive, requires function evaluation |
| **Tournament Selection** | Evolutionary algorithms | Biological metaphor | Less statistical rigor |
| **A/B Testing** | Product experiments | Real user feedback | Slow, expensive, limited scale |

## Development & Testing

### Running Tests

```bash
mix test
```

### Compilation

```bash
mix compile
```

### Documentation

```bash
mix docs
```

## License

MIT License - see LICENSE file for details.

## Credit

This implementation is inspired by and derived from the [RULER (Robust Unified Learning Evaluation & Ranking)](https://art.openpipe.ai/fundamentals/ruler) framework originally developed by OpenPipe for content ranking and trajectory analysis.

## Contact

For questions, issues, or contributions:
- GitHub: https://github.com/OpenPipe/ART
- Hex.pm: https://hex.pm/packages/ranktration
