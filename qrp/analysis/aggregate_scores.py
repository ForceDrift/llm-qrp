# modification of https://github.com/voidism/DoLa/blob/main/dola.py

import json

from pathlib import Path #  for file


def loadAndAggregate(resultsPath: str) -> dict[str, float]:
    with open(resultsPath, "r") as f:
        data = json.load(f)
    layerSums: dict[str, float] = {}
    layerCounts: dict[str, int] = {}

    for sampleKey, sampleVal in data.items():
        result = sampleVal["result"]

        for layerKey, score in result.items():
            if layerKey not in layerSums:
                layerSums[layerKey] = 0.0
                layerCounts[layerKey] = 0
            layerSums[layerKey] += score
            layerCounts[layerKey] += 1

    avgScores = {
        layer: layerSums[layer] / layerCounts[layer]
        for layer in layerSums
    }
    return avgScores

def getSortedLayers(avgScores: dict[str, float]) -> list[tuple[int, float]]:
    parsed = []
    for layerKey, score in avgScores.items():
        layerIdx = int(layerKey.split("_")[1])

        parsed.append((layerIdx, score))

    return sorted(parsed, key=lambda x: x[1])



def saveAggregated(avgScores: dict[str, float], sortedLayers: list[tuple[int, float]], outPath: str) -> None:
    Path(outPath).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "layerAvgScores": avgScores,
        "sortedAscending": [[layerIdx, score] for layerIdx, score in sortedLayers],
    }
    with open(outPath, "w") as f:
        json.dump(output, f, indent=2)
    print(outPath)  
if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    path = os.path.join(repo_root, "results", "gsm8k_200_sled_entropy.json")
    outPath = os.path.join(repo_root, "results", "layer_avg_scores.json")

    avgScores = loadAndAggregate(path)
    sortedLayers = getSortedLayers(avgScores)
    saveAggregated(avgScores, sortedLayers, outPath)

    print(" ---- ascedning order ----")
    for layerIdx, score in sortedLayers:
        print(f"layer {layerIdx}: {score}")
