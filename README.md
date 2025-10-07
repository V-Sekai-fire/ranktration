# Ranktration

Rank/compare algorithms, models, or approaches with weighted multi-criteria analysis.

```elixir
def deps do
  [{:ranktration, "~> 0.1.0"}]
end
```

```elixir
trajectories = [
  Ranktration.TrajectoryResult.new("method_a", "bench", %{"speed" => 0.9, "accuracy" => 0.8}),
  Ranktration.TrajectoryResult.new("method_b", "bench", %{"speed" => 0.7, "accuracy" => 0.95})
]

evaluator = Ranktration.RulerCore.new(metric_weights: %{"speed" => 0.6, "accuracy" => 0.4})
result = Ranktration.RulerCore.evaluate_trajectories(evaluator, trajectories, "bench")
```

## API

- `Ranktration.TrajectoryResult.new(name, content, scores)` - Create trajectory
- `Ranktration.RulerCore.new(weights: %{"metric" => weight})` - Create evaluator
- `Ranktration.RulerCore.evaluate_trajectories(evaluator, trajectories, content)` - Get rankings

Examples: sorting algorithms, ML models, compiler optimizations

## How It Works

1. **Collect trajectories** - Gather approaches with measurable metrics
2. **Pairwise battles** - Compare each against every other using weighted scores
3. **Tournament ranking** - Establish final order through competitive analysis
4. **Statistical confidence** - Measure ranking stability and significance
5. **Final scoring** - Apply ranking bonuses to create comprehensive evaluation

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

Performance: O(n²) pairwise comparisons, optimal for 2-100 trajectories. Sub-millisecond evaluation.

## Development & Testing

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to ensure consistent code formatting and quality.

#### Installation

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install
```

#### Usage

Pre-commit will automatically:
- Format Elixir code with `mix format` on each commit
- Check compilation with `mix compile --warnings-as-errors`

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
- GitHub: https://github.com/V-Sekai-fire/ranktration
- Hex.pm: https://hex.pm/packages/ranktration
