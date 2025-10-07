defmodule Ranktration.MixProject do
  use Mix.Project

  def project do
    [
      app: :ranktration,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "RULER: Robust Unified Learning Evaluation & Ranking",
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    []
  end

  def package do
    [
      name: "ranktration",
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/v-sekai-fire/ranktration"},
      maintainers: ["K. S. Ernest (iFire) Lee"]
    ]
  end
end
