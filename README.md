# Ranktration

Rank/compare algorithms, models, or approaches with weighted multi-criteria analysis.

## Installation

Add to your `mix.exs`:

```elixir
def deps do
  [{:ranktration, "~> 0.1.0"}]
end
```

Then run `mix deps.get`.

## Documentation

Complete API documentation with interactive examples is available via `mix docs`.

## Core Concepts

Ranktration evaluates multiple approaches (trajectories) across multiple criteria using:

1. **Weighted scoring** - Combine multiple quality metrics
2. **Pairwise comparisons** - Tournament-style ranking
3. **Statistical confidence** - Measure ranking reliability
4. **Scalable sampling** - Handle millions of trajectories efficiently

## How It Works

1. **Collect trajectories** - Gather approaches with measurable metrics
2. **Smart sampling** - Select representative sample for large datasets (millions scale)
3. **Pairwise battles** - Compare sample trajectories using weighted scores
4. **Tournament ranking** - Establish global rankings through competitive analysis
5. **Statistical confidence** - Measure ranking stability and significance
6. **Final scoring** - Apply ranking bonuses to create comprehensive evaluation

## Scalability

RULER scales to millions of trajectories using intelligent sampling:

- **Linear scalability**: O(k²) complexity where k = sample_size (vs O(n²) for all n trajectories)
- **Configurable accuracy**: Trade speed vs precision with `sample_size` parameter
- **Massive datasets**: Handle 1M+ trajectories by comparing only hundreds

**Performance modes:**
- `sample_size: 100` → Fast exploration (10k comparisons)
- `sample_size: 1000` → Balanced quality (1M comparisons)
- `sample_size: 5000` → High confidence (25M comparisons)

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

Performance: O(k²) comparisons where k=sample_size, scales to millions of trajectories. Configurable speed vs accuracy trade-offs.

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

This implementation is inspired by and derived from the [RULER (Robust Unified Learning Evaluation & Ranking)](https://art.openpipe.ai/fundamentals/ruler) framework originally developed by OpenPipe for AI evaluation and trajectory analysis in machine learning.

## Contact

For questions, issues, or contributions:
- GitHub: https://github.com/V-Sekai-fire/ranktration
- Hex.pm: https://hex.pm/packages/ranktration
