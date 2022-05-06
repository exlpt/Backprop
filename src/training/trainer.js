import * as tUtil from "./trainingUtil.js";
import * as util from "../util.js";
import * as derivatives from "./derivatives.js";

export function computeGradient(network, batch) {
  // Init out
  let out = {
    weights: [],
    biases: [],
  };
  network.weights.forEach((layer) => {
    let outLayer = [];
    layer.forEach((row) => {
      let outRow = [];
      row.forEach(() => outRow.push(0));
      outLayer.push(outRow);
    });
    out.weights.push(outLayer);
  });

  network.biases.forEach((layer) => {
    let outLayer = [];
    layer.forEach(() => outLayer.push(0));
    out.biases.push(outLayer);
  });

  // Add gradients
  batch.forEach((sample, i) => {
    //console.log("Computing gradient for sample", i, "/", batch.length - 1, "in batch...");
    const sampleGradient = computeSampleGradient(network, sample);

    out.weights.forEach((layer, layerInd) =>
      layer.forEach((row, rowInd) =>
        row.forEach(
          (weight, weightInd) => (row[weightInd] += sampleGradient.weights[layerInd][rowInd][weightInd])
        )
      )
    );

    out.biases.forEach((layer, layerInd) =>
      layer.forEach((bias, biasInd) => (layer[biasInd] += sampleGradient.biases[layerInd][biasInd]))
    );
  });

  // Average gradients
  return {
    weights: out.weights.map((layer) => layer.map((row) => row.map((weight) => weight / batch.length))),
    biases: out.biases.map((layer) => layer.map((bias) => bias / batch.length)),
  };
}

export function computeSampleGradient(network, sample) {
  network.calculate(sample.input);

  // Calc dCost / dOutput (with softmax)
  const netOut = network.layers[network.layers.length - 1];
  // const softOut = util.softmax(netOut);
  // const expSum = netOut.reduce((total, output) => Math.exp(output));
  // const sum = netOut.reduce((total, output) => output);
  // let dLayerAs = netOut.map((output, j) =>
  //   netOut.reduce((total, cur, i) => {
  //     // prettier-ignore
  //     return total
  // 			// dSoftmaxOut / dNetOut
  // 			//(i === j ? 1 : -1) * () / Math.pow(, 2)
  // 			// dCost / dSoftmaxOut
  //   }, 0)
  // );
  let dLayerAs = [];
  dLayerAs[network.layers.length - 1] = netOut.map((output, j) =>
    netOut.reduce((total, cur, i) => total + (2 * (netOut[j] - sample.label[j])) / netOut.length, 0)
  );

  // Backprop through rest of hidden layers
  for (let l = network.layers.length - 2; l > 0; l--) {
    dLayerAs[l] = network.layers[l].map((a, i) =>
      derivatives.dCost_dA(i, network.weights[l], network.layerZs[l], dLayerAs[l + 1])
    );
  }

  // Calc dCost / dWeight
  let gradient = { weights: [], biases: [] };
  for (let l = 0; l < network.weights.length; l++) {
    let layer = [];
    for (let i = 0; i < network.weights[l].length; i++) {
      let row = [];
      for (let j = 0; j < network.weights[l][i].length; j++) {
        row.push(derivatives.dA_dWeight(i, j, network.layers[l], network.layerZs[l]) * dLayerAs[l + 1][i]);
      }
      layer.push(row);
    }
    gradient.weights.push(layer);
  }

  // Calc dCost / dBias
  for (let l = 0; l < network.weights.length; l++) {
    let layer = [];
    for (let i = 0; i < network.weights[l].length; i++) {
      layer.push(derivatives.dA_dBias(i, network.layerZs[l]) * dLayerAs[l + 1][i]);
    }
    gradient.biases.push(layer);
  }

  return gradient;
}

export function applyGradient(network, gradient, scalar = 1) {
  gradient.weights.forEach((layer, layerInd) =>
    layer.forEach((row, rowInd) =>
      row.forEach(
        (weightNudge, colInd) => (network.weights[layerInd][rowInd][colInd] -= weightNudge * scalar)
      )
    )
  );

  gradient.biases.forEach((layer, layerInd) =>
    layer.forEach((biasNudge, biasInd) => (network.biases[layerInd][biasInd] -= biasNudge))
  );
}

export function generateBatch(trainingData, batchSize) {
  if (trainingData.length < batchSize) {
    console.log("function generateBatch => Batch size too large!");
    return null;
  }

  let out = [];
  let occInd = [];
  for (let i = 0; i < batchSize; i++) {
    let desiredIndex = ~~(Math.random() * trainingData.length);
    while (occInd.some((ind) => desiredIndex === ind)) desiredIndex = ~~(Math.random() * trainingData.length);
    occInd.push(desiredIndex);
    out.push(trainingData[desiredIndex]);
  }

  return out;
}
