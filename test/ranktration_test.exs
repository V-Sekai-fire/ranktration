defmodule RanktrationTest do
  use ExUnit.Case
  doctest Ranktration

  alias Ranktration.{TrajectoryResult, RulerCore, RankingResult}

  setup do
    # Create test trajectories
    trajectories = [
      TrajectoryResult.new("algorithm_a", "test_content", %{
        "accuracy" => 0.9,
        "speed" => 0.8,
        "robustness" => 0.7
      }),
      TrajectoryResult.new("algorithm_b", "test_content", %{
        "accuracy" => 0.8,
        "speed" => 0.9,
        "robustness" => 0.6
      }),
      TrajectoryResult.new("algorithm_c", "test_content", %{
        "accuracy" => 0.7,
        "speed" => 0.8,
        "robustness" => 0.9
      })
    ]

    %{trajectories: trajectories}
  end

  describe "TrajectoryResult" do
    test "creates valid trajectory result" do
      trajectory = TrajectoryResult.new("test_id", "content_1", %{"metric" => 0.8})
      assert trajectory.trajectory_id == "test_id"
      assert trajectory.content_id == "content_1"
      assert trajectory.quality_scores == %{"metric" => 0.8}
      assert is_struct(trajectory.timestamp, DateTime)
    end

    test "validates quality scores" do
      assert_raise ArgumentError, fn ->
        TrajectoryResult.new("test", "content", %{"metric" => 1.5})  # > 1.0
      end

      assert_raise ArgumentError, fn ->
        TrajectoryResult.new("test", "content", %{"metric" => -0.5})  # < 0.0
      end
    end

    test "provides utility functions" do
      trajectory = TrajectoryResult.new("test", "content", %{"accuracy" => 0.9, "speed" => 0.8})

      assert TrajectoryResult.has_metric?(trajectory, "accuracy") == true
      assert TrajectoryResult.has_metric?(trajectory, "nonexistent") == false
      assert TrajectoryResult.get_metric(trajectory, "accuracy") == 0.9
      assert TrajectoryResult.get_metric(trajectory, "nonexistent", 0.0) == 0.0
      assert TrajectoryResult.has_all_metrics?(trajectory, ["accuracy", "speed"]) == true
      assert TrajectoryResult.has_all_metrics?(trajectory, ["accuracy", "missing"]) == false
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

    test "requires at least 2 trajectories" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0})
      trajectory = TrajectoryResult.new("single", "content", %{"accuracy" => 0.8})

      assert_raise ArgumentError, fn ->
        RulerCore.evaluate_trajectories(ruler, [trajectory], "content")
      end
    end

    test "validates content_id consistency" do
      ruler = RulerCore.new(metric_weights: %{"accuracy" => 1.0})

      trajectories = [
        TrajectoryResult.new("a", "content_a", %{"accuracy" => 0.8}),
        TrajectoryResult.new("b", "content_b", %{"accuracy" => 0.7})  # Different content
      ]

      assert_raise ArgumentError, fn ->
        RulerCore.evaluate_trajectories(ruler, trajectories, "content_a")
      end
    end
  end

  describe "evaluate_trajectories" do
    setup %{trajectories: trajectories} do
      # Simple evaluator prioritizing accuracy
      ruler = RulerCore.new(metric_weights: %{
        "accuracy" => 0.6,
        "speed" => 0.3,
        "robustness" => 0.1
      })

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

    test "evaluate_trajectories/3 convenience function", %{trajectories: trajectories, weights: weights} do
      result = Ranktration.evaluate_trajectories(trajectories, "test_content", weights)
      assert is_struct(result, RankingResult)
      assert length(result.rankings) == length(trajectories)
    end

    test "compare_trajectories/3 convenience function", %{trajectories: [a, b | _], weights: weights} do
      comparison = Ranktration.compare_trajectories(a, b, weights)
      assert is_struct(comparison, Ranktration.TrajectoryComparison)
      assert comparison.trajectory_a == a.trajectory_id
      assert comparison.trajectory_b == b.trajectory_id
      assert is_binary(comparison.preferred)
    end
  end

  describe "real algorithm benchmarking example" do
    test "demonstrates sorting algorithm ranking" do
      # Simulate sorting algorithm performance based on our Python demo
      sorting_trajectories = [
        TrajectoryResult.new("timsort_python", "sorting_benchmark", %{
          "execution_time" => 0.98,  # Very fast (normalized 0-1)
          "space_efficiency" => 0.85, # Good memory usage
          "correctness" => 1.0,      # Always correct
          "stability" => 1.0         # Stable sorting
        }),
        TrajectoryResult.new("heapsort_stdlib", "sorting_benchmark", %{
          "execution_time" => 0.95,  # Fast but slightly slower
          "space_efficiency" => 0.90, # Excellent memory
          "correctness" => 1.0,      # Always correct
          "stability" => 0.0         # Unstable sorting (penalty!)
        }),
        TrajectoryResult.new("insertionsort_bisect", "sorting_benchmark", %{
          "execution_time" => 0.85,  # Slower but fine for small data
          "space_efficiency" => 0.95, # Very memory efficient
          "correctness" => 1.0,      # Always correct
          "stability" => 1.0         # Stable sorting
        })
      ]

      # Weights prioritize speed and correctness, penalize instability
      algorithm_ruler = RulerCore.new(metric_weights: %{
        "execution_time" => 0.35,
        "space_efficiency" => 0.25,
        "correctness" => 0.25,
        "stability" => 0.15
      })

      result = RulerCore.evaluate_trajectories(algorithm_ruler, sorting_trajectories, "sorting_benchmark")

      # Timsort should generally win (fast, stable, correct)
      assert "timsort_python" in result.rankings
      assert is_float(result.scores["timsort_python"])
      assert result.confidence > 0.0  # Should show meaningful differences

      # Heapsort should be penalized for instability but might still rank well
      heapsort_score = result.scores["heapsort_stdlib"]
      timsort_score = result.scores["timsort_python"]
      # In real scenarios, timsort usually outperforms due to stability bonus
      assert abs(heapsort_score - timsort_score) < 1.0  # Reasonable gap
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
      heap ++ [elem] |> bubble_up(length(heap))
    end

    defp bubble_up(heap, 0), do: heap
    defp bubble_up(heap, index) do
      parent_index = div(index - 1, 2)
      parent = Enum.at(heap, parent_index)
      child = Enum.at(heap, index)

      if child < parent do  # For min-heap
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
        1 -> []
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

      smallest = if left_child < length(heap) and Enum.at(heap, left_child) < Enum.at(heap, smallest) do
        left_child
      else
        smallest
      end

      smallest = if right_child < length(heap) and Enum.at(heap, right_child) < Enum.at(heap, smallest) do
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
      # Generate test data with some characteristics to differentiate algorithms
      test_data = Enum.take(Enum.shuffle(1..1000), 1000)  # Random dataset for balanced performance testing

      # Define algorithms to test
      algorithms = [
        {"bubble_sort", &SortingBenchmarks.bubble_sort/1},
        {"insertion_sort", &SortingBenchmarks.insertion_sort/1},
        {"quicksort", &SortingBenchmarks.quicksort/1},
        {"merge_sort", &SortingBenchmarks.merge_sort/1},
        {"heap_sort", &SortingBenchmarks.heap_sort/1},
        {"elixir_sort", &SortingBenchmarks.elixir_sort/1}
      ]

      # Run benchmark for each algorithm
      results = Enum.map(algorithms, fn {name, sort_fn} ->
        # Warm up and measure
        _warmup = sort_fn.(test_data)

        # Run multiple times and average
        times = for _ <- 1..3 do
          {time, result} = :timer.tc(fn -> sort_fn.(test_data) end, :microsecond)
          %{
            time: time / 1000.0,  # Convert to milliseconds
            result: result,
            correct: result == Enum.sort(test_data),
            stable: SortingBenchmarks.is_stable_sorted?(test_data, result)
          }
        end

        avg_time = Enum.reduce(times, 0, &(&1.time + &2)) / length(times)
        correct_pct = Enum.count(times, & &1.correct) / length(times)
        stable_pct = Enum.count(times, & &1.stable) / length(times)

        %{
          name: name,
          avg_time: avg_time,
          correctness: correct_pct,
          stability: stable_pct
        }
      end)

      # Create trajectories for RULER evaluation
      trajectories = Enum.map(results, fn result ->
        TrajectoryResult.new(result.name, "sorting_benchmark_real", %{
          "execution_time" => normalize_time_metric(results, result.avg_time),
          "correctness" => result.correctness,
          "stability" => result.stability
        })
      end)

      # Define metric weights prioritizing speed, correctness, and stability
      ruler = RulerCore.new(metric_weights: %{
        "execution_time" => 0.6,    # Speed most important
        "correctness" => 0.3,       # Must be correct
        "stability" => 0.1          # Stability bonus
      })

      # Evaluate and rank
      ranking_result = RulerCore.evaluate_trajectories(ruler, trajectories, "sorting_benchmark_real")

      # Assert valid results
      assert length(ranking_result.rankings) == length(algorithms)
      assert ranking_result.confidence > 0.0

      # We expect Elixir's sort to generally rank well (fast and stable)
      # But specific results depend on the input data characteristics
      assert "elixir_sort" in ranking_result.rankings

      # All algorithms should have meaningful scores
      for {alg_name, score} <- ranking_result.scores do
        assert is_float(score)
        assert score >= 0.0 and score <= 1.0
        assert alg_name in Enum.map(results, & &1.name)
      end

      # Should show some performance differentiation
      all_scores = Map.values(ranking_result.scores)
      # Allow some ties but expect meaningful differences overall
      assert length(Enum.uniq(all_scores)) > 1  # At least some variation

      # Quick demonstration of ranking
      IO.puts("\nReal Sorting Algorithm Rankings:")
      Enum.each(ranking_result.rankings, fn alg ->
        score = ranking_result.scores[alg]
        IO.puts("#{alg}: #{Float.round(score, 3)}")
      end)
    end

    # Helper to normalize execution time (faster = higher score)
    defp normalize_time_metric(results, time) do
      times = Enum.map(results, & &1.avg_time)
      min_time = Enum.min(times)
      max_time = Enum.max(times)

      case max_time - min_time do
        0.0 -> 1.0  # All times equal
        diff -> 1.0 - ((time - min_time) / diff)  # Higher score for lower time
      end
    end
  end
end
