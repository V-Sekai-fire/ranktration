defmodule Ranktration.MixProject do
  use Mix.Project

  def project do
    [
      app: :ranktration,
      version: "0.1.1",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "Rank/compare algorithms, models, or approaches with weighted multi-criteria analysis",
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false},
      {:excoveralls, "~> 0.18", only: [:test]}
    ]
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
