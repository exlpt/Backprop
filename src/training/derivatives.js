import { sigmoid } from "../util.js";

export function dCost_dA(j, layerWeights, nextLayerZ, dCost_dNextLayer) {
  return dCost_dNextLayer.reduce(
    (total, dCost_dNextA, i) => total + dA_dA(i, j, layerWeights, nextLayerZ) * dCost_dNextA,
    0
  );
}

export function dA_dA(i, j, layerWeights, nextLayerZ) {
  return layerWeights[i][j] * (sigmoid(nextLayerZ[i]) * (1 - sigmoid(nextLayerZ[i])));
}

export function dA_dWeight(i, j, layerA, nextLayerZ) {
  return layerA[j] * (sigmoid(nextLayerZ[i]) * (1 - sigmoid(nextLayerZ[i])));
}

export function dA_dBias(i, nextLayerZ) {
  return sigmoid(nextLayerZ[i]) * (1 - sigmoid(nextLayerZ[i]));
}
