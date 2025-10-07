defmodule Ranktration do
  @moduledoc """
  RULER - Robust Unified Learning Evaluation & Ranking

  A generalized trajectory evaluation and ranking framework that can work
  across any domain with quantifiable metrics. Originally developed by OpenPipe
  for AI evaluation and trajectory analysis in machine learning.

  ## Usage

      # Create trajectories with domain-specific metrics
      trajectories = [
        Ranktration.TrajectoryResult.new("method_a", "content_1", %{
          "accuracy" => 0.9,
          "speed" => 0.8,
          "robustness" => 0.6
        }),
        Ranktration.TrajectoryResult.new("method_b", "content_1", %{
          "accuracy" => 0.7,
          "speed" => 0.9,
          "robustness" => 0.8
        })
      ]

      # Configure metric weights for your domain
      evaluator = Ranktration.RulerCore.new(metric_weights: %{
        "accuracy" => 0.5,
        "speed" => 0.3,
        "robustness" => 0.2
      })

      # Evaluate and rank
      result = Ranktration.RulerCore.evaluate_trajectories(evaluator, trajectories, "content_1")

      # Access results
      result.rankings  # Ordered list of trajectory IDs
      result.scores    # Map of trajectory_id => final_score
      result.confidence # Overall confidence in ranking

  ## Domain Independence

  RULER works with any domain that has:
  - Multiple trajectories (attempts/solutions)
  - Quantifiable quality metrics
  - Same content being evaluated

  Examples:
  - Algorithm performance evaluation
  - Machine learning model comparisons
  - Image generation quality rankings
  - Compiler optimization comparisons
  """

  defmodule Metrics do
    @moduledoc """
    Structured metrics with compile-time validation for type safety.

    ## Common Metrics
    - `speed`: Execution speed (higher = better)
    - `accuracy`: Accuracy/correctness (higher = better)
    - `correctness`: Boolean accuracy (0.0 or 1.0)
    - `stability`: Algorithm stability (higher = better)
    - `execution_time`: Raw execution time (higher = slower, converted internally)
    - `space_efficiency`: Memory efficiency (higher = better)

    ## Custom Metrics
    Use the `custom` field for domain-specific metrics as `%{:your_metric => value}`.
    """

    @type t :: %__MODULE__{
      speed: float() | nil,
      accuracy: float() | nil,
      correctness: float() | nil,
      stability: float() | nil,
      execution_time: float() | nil,
      space_efficiency: float() | nil,
      custom: %{atom() => float()} | nil
    }

    defstruct [
      :speed,
      :accuracy,
      :correctness,
      :stability,
      :execution_time,
      :space_efficiency,
      custom: %{}
    ]

    @spec new(keyword() | map()) :: t()
    def new(attrs \\ []) do
      struct!(__MODULE__, attrs)
    end

    @spec validate(t()) :: :ok | {:error, String.t()}
    def validate(%__MODULE__{} = metrics) do
      # Check all numeric fields are valid 0.0-1.0 floats
      fields_to_check = [:speed, :accuracy, :correctness, :stability, :execution_time, :space_efficiency]

      Enum.each(fields_to_check, fn field ->
        value = Map.get(metrics, field)
        validate_metric_value(field, value)
      end)

      # Check custom metrics
      if metrics.custom do
        Enum.each(metrics.custom, fn {key, value} ->
          unless is_atom(key) do
            raise ArgumentError, "Custom metric keys must be atoms, got: #{inspect(key)}"
          end
          validate_metric_value(key, value)
        end)
      end

      :ok
    end

    @spec get(t(), atom(), float()) :: float()
    def get(%__MODULE__{} = metrics, key, default \\ 0.0) do
      Map.get(metrics, key) || Map.get(metrics.custom || %{}, key, default)
    end

    @spec has_metric?(t(), atom()) :: boolean()
    def has_metric?(%__MODULE__{} = metrics, key) do
      Map.has_key?(metrics, key) || Map.has_key?(metrics.custom || %{}, key)
    end

    @spec to_map(t()) :: %{String.t() => float()}
    def to_map(%__MODULE__{} = metrics) do
      # Convert struct to string-keyed map for backward compatibility
      metrics
      |> Map.from_struct()
      |> Map.delete(:__struct__)
      |> Map.delete(:custom)
      |> Map.merge(metrics.custom || %{})
      |> Enum.filter(fn {_k, v} -> v != nil end)
      |> Map.new(fn {k, v} -> {Atom.to_string(k), v} end)
    end

    @spec validate_metric_value(atom() | String.t(), any()) :: :ok
    defp validate_metric_value(name, value) do
      unless value == nil or (is_float(value) and value >= 0.0 and value <= 1.0) do
        raise ArgumentError, "Metric #{inspect(name)} must be a float between 0.0 and 1.0, got: #{inspect(value)}"
      end
      :ok
    end
  end

  defmodule TrajectoryResult do
    @moduledoc """
    Structure representing a completed trajectory with quality scores.

    ## Fields
    - `trajectory_id`: Unique identifier for this trajectory
    - `content_id`: Identifier for the content being evaluated
    - `quality_scores`: Map of metric_name => score (0.0-1.0)
    - `metadata`: Optional additional information
    - `timestamp`: When the trajectory was created
    """

    @type t :: %__MODULE__{
            trajectory_id: String.t(),
            content_id: String.t(),
            quality_scores: %{String.t() => float()},
            metadata: map(),
            timestamp: DateTime.t()
          }

    defstruct [
      :trajectory_id,
      :content_id,
      :quality_scores,
      :metadata,
      :timestamp
    ]

    @spec new(String.t(), String.t(), %{String.t() => float()} | Metrics.t(), map()) :: t()
    def new(trajectory_id, content_id, quality_scores, metadata \\ %{}) do
      quality_scores_map = case quality_scores do
        %Metrics{} = struct ->
          Metrics.validate(struct)  # Validate the struct
          Metrics.to_map(struct)   # Convert to string-based map for internal use
        %{} = map ->
          validate_scores(map)
        _ ->
          raise ArgumentError, "quality_scores must be a Metrics struct or a string-keyed map"
      end

      %__MODULE__{
        trajectory_id: trajectory_id,
        content_id: content_id,
        quality_scores: quality_scores_map,
        metadata: metadata,
        timestamp: DateTime.utc_now()
      }
    end

    @spec validate_scores(%{String.t() => float()}) :: %{String.t() => float()}
    defp validate_scores(scores) do
      Enum.each(scores, fn
        {key, value} when is_binary(key) and is_float(value) and value >= 0.0 and value <= 1.0 ->
          :ok

        {key, value} ->
          raise ArgumentError, "Invalid score #{inspect(value)} for metric #{inspect(key)}"
      end)

      scores
    end

    @spec has_metric?(t(), String.t()) :: boolean()
    def has_metric?(%__MODULE__{quality_scores: scores}, metric),
      do: Map.has_key?(scores, metric)

    @spec get_metric(t(), String.t(), float()) :: float()
    def get_metric(%__MODULE__{quality_scores: scores}, metric, default \\ 0.0),
      do: Map.get(scores, metric, default)

    @spec has_all_metrics?(t(), [String.t()]) :: boolean()
    def has_all_metrics?(%__MODULE__{quality_scores: scores}, metrics) do
      Enum.all?(metrics, &Map.has_key?(scores, &1))
    end
  end

  defmodule TrajectoryComparison do
    @moduledoc """
    Contains the result of comparing two trajectories including
    which trajectory is preferred and how confident we are.
    """

    @type t :: %__MODULE__{
            trajectory_a: String.t(),
            trajectory_b: String.t(),
            preferred: String.t(),
            confidence: float(),
            margin: float(),
            quality_differences: %{String.t() => float()}
          }

    defstruct [
      :trajectory_a,
      :trajectory_b,
      :preferred,
      :confidence,
      :margin,
      :quality_differences
    ]
  end

  defmodule RankingResult do
    @moduledoc """
    Complete evaluation result with rankings, scores, and analysis.
    """

    @type t :: %__MODULE__{
            rankings: [String.t()],
            scores: %{String.t() => float()},
            confidence: float(),
            pairwise_comparisons: [TrajectoryComparison.t()],
            consensus_metrics: map(),
            analyzed_at: DateTime.t()
          }

    defstruct [
      :rankings,
      :scores,
      :confidence,
      :pairwise_comparisons,
      :consensus_metrics,
      :analyzed_at
    ]

    @spec top_trajectory(t()) :: String.t() | nil
    def top_trajectory(%__MODULE__{rankings: [first | _]}), do: first

    @spec bottom_trajectory(t()) :: String.t() | nil
    def bottom_trajectory(%__MODULE__{rankings: rankings}),
      do: List.last(rankings)

    @spec get_score(t(), String.t()) :: float() | nil
    def get_score(%__MODULE__{scores: scores}, trajectory_id),
      do: Map.get(scores, trajectory_id)

    @spec get_rank(t(), String.t()) :: pos_integer() | nil
    def get_rank(%__MODULE__{rankings: rankings}, trajectory_id) do
      Enum.find_index(rankings, &(&1 == trajectory_id))
      |> case do
        nil -> nil
        index -> index + 1
      end
    end
  end

  defmodule RulerCore do
    @moduledoc """
    The main RULER evaluation engine that can be configured for different domains.

    Configure it once with your domain's metric weights, then use it to evaluate
    any number of trajectories in that domain.
    """

    @type t :: %__MODULE__{
            metric_weights: %{String.t() => float()},
            config: map()
          }

    defstruct [:metric_weights, :config]

    @spec new(keyword()) :: t()
    def new(opts \\ []) do
      metric_weights = Keyword.get(opts, :metric_weights, %{})
      validate_metric_weights!(metric_weights)

      %__MODULE__{
        metric_weights: metric_weights,
        config: Map.new(opts)
      }
    end

    @spec configure_metrics(t(), %{String.t() => float()}) :: t()
    def configure_metrics(%__MODULE__{} = ruler, new_weights) do
      validate_metric_weights!(new_weights)
      %{ruler | metric_weights: new_weights}
    end

    @spec evaluate_trajectories(t(), [TrajectoryResult.t()], String.t()) :: RankingResult.t()
    def evaluate_trajectories(%__MODULE__{} = ruler, trajectories, content_id) do
      if length(trajectories) < 2 do
        raise ArgumentError, "Need at least 2 trajectories to rank, got #{length(trajectories)}"
      end

      # Validate all trajectories are for the same content
      Enum.each(trajectories, fn traj ->
        if traj.content_id != content_id do
          raise ArgumentError,
                "All trajectories must be for the same content #{inspect(content_id)}"
        end
      end)

      # Perform pairwise comparisons
      pairwise_comparisons = compare_all_trajectories(ruler, trajectories)

      # Calculate consensus ranking
      rankings = calculate_consensus_ranking(pairwise_comparisons)

      # Calculate final scores with ranking bonuses
      scores = calculate_final_scores(ruler, trajectories, pairwise_comparisons)

      # Analyze consensus and stability
      consensus_metrics = analyze_consensus(pairwise_comparisons)

      # Calculate overall confidence
      confidence = calculate_overall_confidence(pairwise_comparisons)

      %RankingResult{
        rankings: rankings,
        scores: scores,
        confidence: confidence,
        pairwise_comparisons: pairwise_comparisons,
        consensus_metrics: consensus_metrics,
        analyzed_at: DateTime.utc_now()
      }
    end

    # Private implementation details extracted from our Python RULER
    @spec compare_all_trajectories(t(), [TrajectoryResult.t()]) :: [TrajectoryComparison.t()]
    defp compare_all_trajectories(ruler, trajectories) do
      for a <- trajectories, b <- trajectories, a.trajectory_id < b.trajectory_id do
        compare_trajectory_pair(ruler, a, b)
      end
    end

    @spec compare_trajectory_pair(t(), TrajectoryResult.t(), TrajectoryResult.t()) ::
            TrajectoryComparison.t()
    def compare_trajectory_pair(ruler, %TrajectoryResult{} = a, %TrajectoryResult{} = b) do
      # Calculate weighted scores for each trajectory
      score_a = calculate_weighted_score(ruler, a.quality_scores)
      score_b = calculate_weighted_score(ruler, b.quality_scores)

      # Calculate quality differences for analysis
      quality_differences =
        calculate_quality_differences(ruler.metric_weights, a.quality_scores, b.quality_scores)

      # Determine preferred trajectory
      {preferred, margin} =
        if abs(score_a - score_b) < 1.0e-6 do
          # Tie-breaking logic when scores are equal
          {compare_tie_breaker(ruler, a, b), 0.0}
        else
          preferred = if score_a > score_b, do: a.trajectory_id, else: b.trajectory_id
          margin = abs(score_a - score_b)
          {preferred, margin}
        end

      # Calculate confidence based on margin and variance
      # Simplified confidence calculation
      confidence = min(1.0, margin * 2.0)

      %TrajectoryComparison{
        trajectory_a: a.trajectory_id,
        trajectory_b: b.trajectory_id,
        preferred: preferred,
        confidence: confidence,
        margin: margin,
        quality_differences: quality_differences
      }
    end

    @spec calculate_weighted_score(t(), %{String.t() => float()}) :: float()
    defp calculate_weighted_score(ruler, scores) do
      Enum.reduce(ruler.metric_weights, 0.0, fn {metric, weight}, acc ->
        score = Map.get(scores, metric, 0.0)
        acc + score * weight
      end)
    end

    @spec calculate_quality_differences(%{String.t() => float()}, %{String.t() => float()}, %{
            String.t() => float()
          }) :: %{String.t() => float()}
    defp calculate_quality_differences(weights, scores_a, scores_b) do
      Map.new(weights, fn {metric, _weight} ->
        a_score = Map.get(scores_a, metric, 0.0)
        b_score = Map.get(scores_b, metric, 0.0)
        {metric, a_score - b_score}
      end)
    end

    @spec compare_tie_breaker(t(), TrajectoryResult.t(), TrajectoryResult.t()) :: String.t()
    defp compare_tie_breaker(_ruler, a, b) do
      # Simple alphabetical tie-breaker for stability
      if a.trajectory_id <= b.trajectory_id do
        a.trajectory_id
      else
        b.trajectory_id
      end
    end

    @spec calculate_consensus_ranking([TrajectoryComparison.t()]) :: [String.t()]
    defp calculate_consensus_ranking(comparisons) do
      # Count wins for each trajectory
      win_counts =
        Enum.reduce(comparisons, %{}, fn comp, acc ->
          winner = comp.preferred
          Map.update(acc, winner, 1, &(&1 + 1))
        end)

      # Debug: IO.puts("Win counts: #{inspect win_counts}")

      # Ensure all trajectories are included (even if they have 0 wins)
      all_trajectory_ids =
        comparisons
        |> Enum.flat_map(fn comp -> [comp.trajectory_a, comp.trajectory_b] end)
        |> Enum.uniq()

      # Initialize 0 wins for trajectories not yet in win_counts
      complete_win_counts =
        Enum.reduce(all_trajectory_ids, win_counts, fn id, acc ->
          Map.put_new(acc, id, 0)
        end)

      # Debug: IO.puts("Complete win counts: #{inspect complete_win_counts}")

      # Sort by win counts (descending), then alphabetically for ties
      complete_win_counts
      |> Enum.sort_by(fn {id, wins} -> {-wins, id} end)
      |> Enum.map(fn {id, _wins} -> id end)
    end

    @spec calculate_final_scores(t(), [TrajectoryResult.t()], [TrajectoryComparison.t()]) :: %{
            String.t() => float()
          }
    defp calculate_final_scores(ruler, trajectories, comparisons) do
      trajectory_ids = Enum.map(trajectories, & &1.trajectory_id)
      rankings = calculate_consensus_ranking(comparisons)

      Enum.reduce(trajectory_ids, %{}, fn traj_id, acc ->
        # Base score from weighted metrics
        traj = Enum.find(trajectories, &(&1.trajectory_id == traj_id))
        base_score = calculate_weighted_score(ruler, traj.quality_scores)

        # Ranking bonus (higher ranking gets small bonus)
        rank_position = Enum.find_index(rankings, &(&1 == traj_id)) || length(rankings)
        ranking_bonus = (length(rankings) - rank_position) / length(rankings) * 0.05

        Map.put(acc, traj_id, min(1.0, base_score + ranking_bonus))
      end)
    end

    @spec analyze_consensus([TrajectoryComparison.t()]) :: map()
    defp analyze_consensus(comparisons) do
      # Calculate win distribution and other consensus metrics
      wins_distribution =
        Enum.reduce(comparisons, %{}, fn comp, acc ->
          Map.update(acc, comp.preferred, 1, &(&1 + 1))
        end)

      confidence_stats = Enum.map(comparisons, & &1.confidence)

      avg_confidence =
        if confidence_stats == [],
          do: 0.0,
          else: Enum.sum(confidence_stats) / length(confidence_stats)

      %{
        "wins_distribution" => wins_distribution,
        "average_confidence" => avg_confidence,
        "total_comparisons" => length(comparisons)
      }
    end

    @spec calculate_overall_confidence([TrajectoryComparison.t()]) :: float()
    defp calculate_overall_confidence(comparisons) do
      if comparisons == [] do
        0.0
      else
        confidences = Enum.map(comparisons, & &1.confidence)
        Enum.sum(confidences) / length(confidences)
      end
    end

    @spec validate_metric_weights!(%{String.t() => float()}) :: :ok
    defp validate_metric_weights!(weights) do
      unless is_map(weights) do
        raise ArgumentError, "metric_weights must be a map"
      end

      total_weight = Enum.sum(Map.values(weights))

      if abs(total_weight - 1.0) > 1.0e-3 do
        raise ArgumentError, "metric weights must sum to 1.0, got #{total_weight}"
      end

      Enum.each(weights, fn
        {key, value} when is_binary(key) and is_float(value) and value >= 0.0 ->
          :ok

        {key, value} ->
          raise ArgumentError, "Invalid weight #{inspect(value)} for metric #{inspect(key)}"
      end)
    end
  end

  # Convenience functions for easy access
  @spec evaluate_trajectories([TrajectoryResult.t()], String.t(), %{String.t() => float()}) ::
          RankingResult.t()
  def evaluate_trajectories(trajectories, content_id, weights) do
    ruler = RulerCore.new(metric_weights: weights)
    RulerCore.evaluate_trajectories(ruler, trajectories, content_id)
  end

  @spec compare_trajectories(TrajectoryResult.t(), TrajectoryResult.t(), %{String.t() => float()}) ::
          TrajectoryComparison.t()
  def compare_trajectories(a, b, weights) do
    ruler = RulerCore.new(metric_weights: weights)
    RulerCore.compare_trajectory_pair(ruler, a, b)
  end
end
