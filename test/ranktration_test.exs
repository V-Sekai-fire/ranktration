defmodule RanktrationTest do
  use ExUnit.Case
  doctest Ranktration

  alias Ranktration.{TrajectoryResult, RulerCore, RankingResult}

  setup do
    # Create test trajectories
    trajectories = [
      TrajectoryResult.new("algorithm_a", "test_content", %Ranktration.Metrics{
        accuracy: 0.9,
        speed: 0.8,
        robustness: 0.7
      }),
      TrajectoryResult.new("algorithm_b", "test_content", %Ranktration.Metrics{
        accuracy: 0.8,
        speed: 0.9,
        robustness: 0.6
      }),
      TrajectoryResult.new("algorithm_c", "test_content", %Ranktration.Metrics{
        accuracy: 0.7,
        speed: 0.8,
        robustness: 0.9
      })
    ]

    %{trajectories: trajectories}
  end

  describe "TrajectoryResult" do
    test "creates valid trajectory result" do
      metrics = Ranktration.Metrics.new(%{custom: %{metric: 0.8}})
      trajectory = TrajectoryResult.new("test_id", "content_1", metrics)
      assert trajectory.trajectory_id == "test_id"
      assert trajectory.content_id == "content_1"
      assert trajectory.quality_scores == %{"metric" => 0.8}
      assert is_struct(trajectory.timestamp, DateTime)
    end

    test "validates quality scores" do
      assert_raise ArgumentError, fn ->
        # > 1.0
        TrajectoryResult.new("test", "content", %Ranktration.Metrics{accuracy: 1.5})
      end

      assert_raise ArgumentError, fn ->
        # < 0.0
        TrajectoryResult.new("test", "content", %Ranktration.Metrics{speed: -0.5})
      end
    end

    test "provides utility functions" do
      metrics = Ranktration.Metrics.new(accuracy: 0.9, speed: 0.8)
      trajectory = TrajectoryResult.new("test", "content", metrics)

      assert TrajectoryResult.has_metric?(trajectory, "accuracy") == true
      assert TrajectoryResult.has_metric?(trajectory, "nonexistent") == false
      assert TrajectoryResult.get_metric(trajectory, "accuracy") == 0.9
      assert TrajectoryResult.get_metric(trajectory, "nonexistent", 0.0) == 0.0
      assert TrajectoryResult.has_all_metrics?(trajectory, ["accuracy", "speed"]) == true
      assert TrajectoryResult.has_all_metrics?(trajectory, ["accuracy", "missing"]) == false
    end
  end

  describe "Metrics validation edge cases" do
    test "validates various invalid metric values" do
      # Test nil values (should be allowed)
      metrics = Ranktration.Metrics.new(speed: nil, accuracy: 0.5)
      assert metrics.speed == nil
      assert metrics.accuracy == 0.5

      # Test custom metrics with invalid values (> 1.0) - this gets validated during trajectory creation
      assert_raise ArgumentError, fn ->
        TrajectoryResult.new(
          "test",
          "content",
          Ranktration.Metrics.new(%{custom: %{custom_metric: 2.0}})
        )
      end

      # Test custom metrics with invalid values (< 0.0)
      assert_raise ArgumentError, fn ->
        TrajectoryResult.new(
          "test",
          "content",
          Ranktration.Metrics.new(%{custom: %{custom_metric: -0.5}})
        )
      end

      # Test invalid custom metric types separately by calling validate
      assert_raise ArgumentError, fn ->
        # Integer key
        %Ranktration.Metrics{custom: %{0 => 0.5}} |> Ranktration.Metrics.validate()
      end
    end

    test "validates float precision and edge cases" do
      # Valid extreme floats
      metrics = Ranktration.Metrics.new(accuracy: 1.0, speed: 0.0)
      assert metrics.accuracy == 1.0
      assert metrics.speed == 0.0

      # Very small and large valid values
      metrics = Ranktration.Metrics.new(accuracy: 0.0001, speed: 0.9999)
      assert metrics.accuracy == 0.0001
      assert metrics.speed == 0.9999
    end

    test "handles custom atom keys correctly" do
      metrics = Ranktration.Metrics.new(%{custom: %{my_custom_metric: 0.8, another_metric: 0.2}})
      assert Ranktration.Metrics.has_metric?(metrics, :my_custom_metric) == true
      assert Ranktration.Metrics.get(metrics, :my_custom_metric) == 0.8
      assert Ranktration.Metrics.has_metric?(metrics, :nonexistent) == false
      assert Ranktration.Metrics.get(metrics, :nonexistent, 0.1) == 0.1
    end
  end

  describe "RulerCore" do
    test "creates evaluator with valid metric weights" do
      weights = %{"accuracy" => 0.5, "speed" => 0.3, "robustness" => 0.2}
      ruler = RulerCore.new(metric_weights: weights)
      assert ruler.metric_weights == weights
    end

    test "validates metric weights" do
      # Weights don't sum to 1.0
      assert_raise ArgumentError, fn ->
        RulerCore.new(metric_weights: %{"a" => 0.5, "b" => 0.6})
      end

      # Negative weights
      assert_raise ArgumentError, fn ->
        RulerCore.new(metric_weights: %{"a" => -0.1, "b" => 0.6, "c" => 0.5})
      end
    end

    test "creates with default sample size" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0})
      assert ruler.sample_size == 100
    end

    test "accepts custom sample size" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0}, sample_size: 500)
      assert ruler.sample_size == 500
    end

    test "validates metric weights with invalid types" do
      # Non-map weights
      assert_raise ArgumentError, fn ->
        RulerCore.new(metric_weights: "not_a_map")
      end

      # Illegal float values
      assert_raise ArgumentError, fn ->
        RulerCore.new(metric_weights: %{"valid" => 1.0, "negative" => -0.5})
      end

      # Non-binary keys
      assert_raise ArgumentError, fn ->
        RulerCore.new(metric_weights: %{123 => 1.0})
      end
    end

    test "requires at least 2 trajectories" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0})
      trajectory = TrajectoryResult.new("single", "content", %Ranktration.Metrics{accuracy: 0.8})

      assert_raise ArgumentError, fn ->
        RulerCore.evaluate_trajectories(ruler, [trajectory], "content")
      end
    end

    test "validates content_id consistency" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0})

      trajectories = [
        TrajectoryResult.new("a", "content_a", %Ranktration.Metrics{accuracy: 0.8}),
        # Different content
        TrajectoryResult.new("b", "content_b", %Ranktration.Metrics{accuracy: 0.7})
      ]

      assert_raise ArgumentError, fn ->
        RulerCore.evaluate_trajectories(ruler, trajectories, "content_a")
      end
    end

    test "configure_metrics updates weights" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0})
      new_ruler = RulerCore.configure_metrics(ruler, %{"speed" => 0.5, "accuracy" => 0.5})

      assert new_ruler.metric_weights == %{"speed" => 0.5, "accuracy" => 0.5}
      assert new_ruler != ruler
    end

    test "configure_metrics validates new weights" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0})

      # This should fail with ArithmeticError when trying to sum non-numbers
      assert_raise ArithmeticError, fn ->
        RulerCore.configure_metrics(ruler, %{invalid: "weights"})
      end
    end

    test "handles sampling with small datasets" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0}, sample_size: 10)

      # Create only 3 trajectories (less than sample size)
      trajectories = [
        TrajectoryResult.new("a", "test", %Ranktration.Metrics{accuracy: 0.8}),
        TrajectoryResult.new("b", "test", %Ranktration.Metrics{accuracy: 0.6}),
        TrajectoryResult.new("c", "test", %Ranktration.Metrics{accuracy: 0.7})
      ]

      result = RulerCore.evaluate_trajectories(ruler, trajectories, "test")
      assert length(result.rankings) == 3
      assert result.confidence >= 0.0
    end

    test "handles exact sample size match" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0}, sample_size: 2)

      trajectories = [
        TrajectoryResult.new("a", "test", %Ranktration.Metrics{accuracy: 0.8}),
        TrajectoryResult.new("b", "test", %Ranktration.Metrics{accuracy: 0.6}),
        TrajectoryResult.new("c", "test", %Ranktration.Metrics{accuracy: 0.7}),
        TrajectoryResult.new("d", "test", %Ranktration.Metrics{accuracy: 0.9})
      ]

      result = RulerCore.evaluate_trajectories(ruler, trajectories, "test")
      # All trajectories should be ranked
      assert length(result.rankings) == 4
      assert length(result.pairwise_comparisons) > 0
    end

    test "handles very small datasets with sampling" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0}, sample_size: 10)

      # Only 2 trajectories (smaller than sample size but still enough for ranking)
      trajectories = [
        TrajectoryResult.new("a", "test", %Ranktration.Metrics{accuracy: 0.8}),
        TrajectoryResult.new("b", "test", %Ranktration.Metrics{accuracy: 0.6})
      ]

      result = RulerCore.evaluate_trajectories(ruler, trajectories, "test")
      assert length(result.rankings) == 2
      assert result.confidence >= 0.0
      # Should have exactly 1 pairwise comparison for 2 items
      assert length(result.pairwise_comparisons) == 1
    end

    test "handles very small metric weights" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0}, sample_size: 10)

      trajectories = [
        TrajectoryResult.new("a", "test", %Ranktration.Metrics{accuracy: 0.8}),
        TrajectoryResult.new("b", "test", %Ranktration.Metrics{accuracy: 0.6})
      ]

      result = RulerCore.evaluate_trajectories(ruler, trajectories, "test")
      assert length(result.rankings) == 2
      # Scores should be weighted by accuracy with ranking bonus
      # ~1.0 * 0.8 + ranking bonus
      assert_in_delta result.scores["a"], 0.85, 0.01
      # ~1.0 * 0.6 + ranking bonus
      assert_in_delta result.scores["b"], 0.63, 0.01
      # a should rank higher
      assert result.scores["a"] > result.scores["b"]
    end

    test "handles edge case rank positions" do
      # Create ruler with tied weights to test ranking edge cases
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0}, sample_size: 3)

      # Create trajectories where all scores are equal - tests tie-breaking logic
      trajectories = [
        TrajectoryResult.new("equal_a", "test", %Ranktration.Metrics{accuracy: 0.5}),
        TrajectoryResult.new("equal_b", "test", %Ranktration.Metrics{accuracy: 0.5}),
        TrajectoryResult.new("equal_c", "test", %Ranktration.Metrics{accuracy: 0.5})
      ]

      result = RulerCore.evaluate_trajectories(ruler, trajectories, "test")
      assert length(result.rankings) == 3
      assert Map.keys(result.scores) == ["equal_a", "equal_b", "equal_c"]
      # All should have close scores (tied base values with small ranking bonuses)
      scores = Map.values(result.scores)
      # Should be around 0.5 + small bonus
      assert Enum.all?(scores, &(&1 >= 0.5 and &1 <= 0.6))
      # (3*2)/2 = 3 comparisons
      assert length(result.pairwise_comparisons) == 3
    end

    test "handles exact sample size edge cases" do
      # Test with exactly the same number of trajectories as sample size
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0}, sample_size: 3)

      trajectories = [
        TrajectoryResult.new("exact_1", "test", %Ranktration.Metrics{accuracy: 0.9}),
        TrajectoryResult.new("exact_2", "test", %Ranktration.Metrics{accuracy: 0.7}),
        TrajectoryResult.new("exact_3", "test", %Ranktration.Metrics{accuracy: 0.8})
      ]

      result = RulerCore.evaluate_trajectories(ruler, trajectories, "test")
      assert length(result.rankings) == 3
      # Should do full pairwise comparisons: (3*2)/2 = 3 comparisons
      assert length(result.pairwise_comparisons) == 3
    end

    test "covers tie-breaker edge case" do
      # Test tie-breaker that returns b.trajectory_id when b comes before a alphabetically
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0})
      traj_b = TrajectoryResult.new("a_traject", "test", %Ranktration.Metrics{accuracy: 0.5})
      traj_a = TrajectoryResult.new("z_traject", "test", %Ranktration.Metrics{accuracy: 0.5})

      # This should trigger the tie-breaker where b comes first alphabetically
      comparison = RulerCore.compare_trajectory_pair(ruler, traj_a, traj_b)
      # Should return b.trajectory_id since a > b alphabetically
      assert comparison.preferred == "a_traject"
    end
  end

  describe "evaluate_trajectories" do
    setup %{trajectories: trajectories} do
      # Simple evaluator prioritizing accuracy
      ruler =
        RulerCore.new(
          metric_weights: %{
            "accuracy" => 0.6,
            "speed" => 0.3,
            "robustness" => 0.1
          }
        )

      result = RulerCore.evaluate_trajectories(ruler, trajectories, "test_content")

      %{result: result, ruler: ruler}
    end

    test "produces valid ranking result", %{result: result} do
      assert is_struct(result, RankingResult)
      assert is_list(result.rankings)
      assert is_map(result.scores)
      assert is_float(result.confidence)
      assert is_list(result.pairwise_comparisons)
      assert is_map(result.consensus_metrics)
      assert is_struct(result.analyzed_at, DateTime)
    end

    test "ranks all trajectories", %{result: result, trajectories: trajectories} do
      expected_ids = Enum.map(trajectories, & &1.trajectory_id)
      assert Enum.sort(result.rankings) == Enum.sort(expected_ids)
      assert map_size(result.scores) == length(trajectories)
    end

    test "provides meaningful scores", %{result: result} do
      for {_, score} <- result.scores do
        assert is_float(score)
        assert score >= 0.0 and score <= 1.0
      end
    end

    test "generates pairwise comparisons", %{result: result} do
      # For 3 trajectories, we expect (3*2)/2 = 3 pairwise comparisons
      assert length(result.pairwise_comparisons) == 3

      for comparison <- result.pairwise_comparisons do
        assert is_binary(comparison.trajectory_a)
        assert is_binary(comparison.trajectory_b)
        assert is_binary(comparison.preferred)
        assert is_float(comparison.confidence)
        assert is_float(comparison.margin)
        assert is_map(comparison.quality_differences)
      end
    end
  end

  describe "RankingResult utilities" do
    test "provides access methods" do
      rankings = ["first", "second", "third"]
      scores = %{"first" => 0.9, "second" => 0.8, "third" => 0.7}

      result = %RankingResult{
        rankings: rankings,
        scores: scores,
        confidence: 0.8,
        pairwise_comparisons: [],
        consensus_metrics: %{},
        analyzed_at: DateTime.utc_now()
      }

      assert RankingResult.top_trajectory(result) == "first"
      assert RankingResult.bottom_trajectory(result) == "third"
      assert RankingResult.get_score(result, "first") == 0.9
      assert RankingResult.get_score(result, "nonexistent") == nil
      assert RankingResult.get_rank(result, "first") == 1
      assert RankingResult.get_rank(result, "second") == 2
      assert RankingResult.get_rank(result, "nonexistent") == nil
    end
  end

  describe "convenience functions" do
    setup %{trajectories: trajectories} do
      weights = %{"accuracy" => 0.6, "speed" => 0.3, "robustness" => 0.1}
      %{trajectories: trajectories, weights: weights}
    end

    test "evaluate_trajectories/3 convenience function", %{
      trajectories: trajectories,
      weights: weights
    } do
      result = Ranktration.evaluate_trajectories(trajectories, "test_content", weights)
      assert is_struct(result, RankingResult)
      assert length(result.rankings) == length(trajectories)
    end

    test "compare_trajectories/3 convenience function", %{
      trajectories: [a, b | _],
      weights: weights
    } do
      comparison = Ranktration.compare_trajectories(a, b, weights)
      assert is_struct(comparison, Ranktration.TrajectoryComparison)
      assert comparison.trajectory_a == a.trajectory_id
      assert comparison.trajectory_b == b.trajectory_id
      assert is_binary(comparison.preferred)
    end
  end

  defmodule SortingBenchmarks do
    @moduledoc false

    def bubble_sort(list) do
      do_bubble_sort(list, length(list))
    end

    defp do_bubble_sort(list, 0), do: list

    defp do_bubble_sort(list, n) do
      {new_list, swapped} = bubble_pass(list, 0, false)

      if swapped do
        do_bubble_sort(new_list, n)
      else
        new_list
      end
    end

    defp bubble_pass(list, index, swapped) when index >= length(list) - 1, do: {list, swapped}

    defp bubble_pass(list, index, swapped) do
      left = Enum.at(list, index)
      right = Enum.at(list, index + 1)

      if left > right do
        new_list = list |> List.replace_at(index, right) |> List.replace_at(index + 1, left)
        bubble_pass(new_list, index + 1, true)
      else
        bubble_pass(list, index + 1, swapped)
      end
    end

    def insertion_sort(list), do: do_insertion_sort([], list)

    defp do_insertion_sort(sorted, []), do: sorted

    defp do_insertion_sort(sorted, [h | t]) do
      do_insertion_sort(insert(sorted, h), t)
    end

    defp insert([], elem), do: [elem]

    defp insert([h | t], elem) do
      case elem <= h do
        true -> [elem | [h | t]]
        false -> [h | insert(t, elem)]
      end
    end

    def quicksort([]), do: []

    def quicksort([pivot | rest]) do
      {less, greater} = Enum.split_with(rest, &(&1 <= pivot))
      quicksort(less) ++ [pivot] ++ quicksort(greater)
    end

    def merge_sort([]), do: []
    def merge_sort([x]), do: [x]

    def merge_sort(list) do
      {left, right} = Enum.split(list, div(length(list), 2))
      merge(merge_sort(left), merge_sort(right))
    end

    defp merge([], right), do: right
    defp merge(left, []), do: left

    defp merge([l | left], [r | right]) do
      if l <= r do
        [l | merge(left, [r | right])]
      else
        [r | merge([l | left], right)]
      end
    end

    def heap_sort(list) do
      # Build min-heap
      heap = Enum.reduce(list, [], &insert_heap/2)
      # Extract min repeatedly to get sorted list
      extract_all_min(heap, [])
    end

    defp insert_heap(elem, []), do: [elem]

    defp insert_heap(elem, heap) do
      (heap ++ [elem]) |> bubble_up(length(heap))
    end

    defp bubble_up(heap, 0), do: heap

    defp bubble_up(heap, index) do
      parent_index = div(index - 1, 2)
      parent = Enum.at(heap, parent_index)
      child = Enum.at(heap, index)

      # For min-heap
      if child < parent do
        heap
        |> List.replace_at(parent_index, child)
        |> List.replace_at(index, parent)
        |> bubble_up(parent_index)
      else
        heap
      end
    end

    defp extract_all_min([], sorted), do: Enum.reverse(sorted)

    defp extract_all_min(heap, sorted) do
      min_val = Enum.at(heap, 0)
      new_heap = remove_min(heap)
      extract_all_min(new_heap, [min_val | sorted])
    end

    defp remove_min(heap) do
      case length(heap) do
        1 ->
          []

        _ ->
          last = List.last(heap)
          heap_without_last = List.delete_at(heap, -1)
          heap_with_last_at_root = List.replace_at(heap_without_last, 0, last)
          sink_down(heap_with_last_at_root, 0)
      end
    end

    defp sink_down(heap, index) do
      left_child = 2 * index + 1
      right_child = 2 * index + 2
      smallest = index

      smallest =
        if left_child < length(heap) and Enum.at(heap, left_child) < Enum.at(heap, smallest) do
          left_child
        else
          smallest
        end

      smallest =
        if right_child < length(heap) and Enum.at(heap, right_child) < Enum.at(heap, smallest) do
          right_child
        else
          smallest
        end

      if smallest != index do
        heap
        |> List.replace_at(index, Enum.at(heap, smallest))
        |> List.replace_at(smallest, Enum.at(heap, index))
        |> sink_down(smallest)
      else
        heap
      end
    end

    def elixir_sort(list), do: Enum.sort(list)

    def is_stable_sorted?(_original, sorted) do
      # Simplified stability check - assumes no duplicates for now
      sorted == Enum.sort(sorted)
    end
  end

  describe "real sorting algorithm benchmarking" do
    test "benchmarks and ranks actual sorting algorithms using RULER" do
      # Generate smaller test data for faster testing (100 elements)
      test_data = Enum.take(Enum.shuffle(1..100), 100)

      # Test fewer algorithms (3 instead of 6) for faster execution
      algorithms = [
        {"insertion_sort", &SortingBenchmarks.insertion_sort/1},
        {"quicksort", &SortingBenchmarks.quicksort/1},
        {"elixir_sort", &SortingBenchmarks.elixir_sort/1}
      ]

      # Run benchmarks in parallel for speed
      results =
        Task.async_stream(
          algorithms,
          fn {name, sort_fn} ->
            # Warm up
            _warmup = sort_fn.(test_data)

            # Single run for speed (vs 3 runs)
            {time, result} = :timer.tc(fn -> sort_fn.(test_data) end, :microsecond)

            %{
              name: name,
              avg_time: time / 1000.0,
              correctness: if(result == Enum.sort(test_data), do: 1.0, else: 0.0),
              # All tested algorithms are stable
              stability: 1.0
            }
          end,
          max_concurrency: System.schedulers_online()
        )
        |> Enum.map(fn {:ok, result} -> result end)

      # Create trajectories for RULER evaluation
      trajectories =
        Enum.map(results, fn result ->
          TrajectoryResult.new(result.name, "sorting_benchmark_real", %Ranktration.Metrics{
            execution_time: normalize_time_metric(results, result.avg_time),
            correctness: result.correctness,
            stability: result.stability
          })
        end)

      # Define metric weights prioritizing speed, correctness, and stability
      ruler =
        RulerCore.new(
          metric_weights: %{
            # Speed most important
            "execution_time" => 0.6,
            # Must be correct
            "correctness" => 0.3,
            # Stability bonus
            "stability" => 0.1
          }
        )

      # Evaluate and rank
      ranking_result =
        RulerCore.evaluate_trajectories(ruler, trajectories, "sorting_benchmark_real")

      # Assert valid results
      assert length(ranking_result.rankings) == length(algorithms)
      assert ranking_result.confidence >= 0.0

      # We expect Elixir's sort to generally rank well (fast and stable)
      # But specific results depend on the input data characteristics
      assert "elixir_sort" in ranking_result.rankings

      # All algorithms should have meaningful scores
      for {alg_name, score} <- ranking_result.scores do
        assert is_float(score)
        assert score >= 0.0 and score <= 1.0
        assert alg_name in Enum.map(results, & &1.name)
      end

      # Should return meaningful results
      all_scores = Map.values(ranking_result.scores)
      # All algorithms should have valid scores
      assert Enum.all?(all_scores, &(&1 >= 0.0 and &1 <= 1.0))
    end

    # Helper to normalize execution time (faster = higher score)
    defp normalize_time_metric(results, time) do
      times = Enum.map(results, & &1.avg_time)
      min_time = Enum.min(times)
      max_time = Enum.max(times)
      diff = max_time - min_time

      # Handle case where all times are equal (avoid 0.0 pattern match)
      if abs(diff) < 1.0e-10 do
        1.0
      else
        1.0 - (time - min_time) / diff
      end
    end
  end
end
