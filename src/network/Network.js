import { sigmoid, dotProd } from "../util.js";

export default class Network {
  constructor(...layerLengths) {
    // Create layers & Zs
    this.layers = [];
    this.layerZs = [];
    layerLengths.forEach((layerLength) => {
      let layer = [];
      for (let i = 0; i < layerLength; i++) layer.push(null);
      this.layers.push(layer);
      this.layerZs.push([...layer]);
    });
    this.layerZs = this.layerZs.slice(1);

    // Create weight matricies
    this.weights = [];
    for (let i = 1; i < layerLengths.length; i++) {
      let layerWeightMatrix = [];
      for (let out = 0; out < layerLengths[i]; out++) {
        let row = [];
        for (let inp = 0; inp < layerLengths[i - 1]; inp++) row.push(null);
        layerWeightMatrix.push(row);
      }
      this.weights.push(layerWeightMatrix);
    }

    // Create biases
    this.biases = [];
    for (let i = 1; i < layerLengths.length; i++) {
      let layerBiases = this.layers[i].map(() => null);
      this.biases.push(layerBiases);
    }
  }

  calculate(input) {
    if (input.length !== this.layers[0].length) {
      if (this.weights[0][0][0] === null) {
        console.log("class Network => Weight matricies uninitialized!");
        return null;
      }
      console.log("class Network => Invalid input length!");
      return null;
    }

    this.layers[0] = [...input];
    for (let l = 1; l < this.layers.length; l++)
      this.layers[l] = this.weights[l - 1].map((weightsRow, nodeInd) => {
        const z = dotProd(this.layers[l - 1], weightsRow) + this.biases[l - 1][nodeInd];
        this.layerZs[l - 1][nodeInd] = z;
        return sigmoid(z);
      });

    return [...this.layers[this.layers.length - 1]];
  }

  initWeights(val = null) {
    for (let l = 0; l < this.weights.length; l++)
      for (let row = 0; row < this.weights[l].length; row++)
        for (let col = 0; col < this.weights[l][row].length; col++)
          this.weights[l][row][col] = val === null ? Math.random() * 2 - 1 : val;
  }

  initBiases(val = null) {
    for (let l = 0; l < this.biases.length; l++)
      for (let bias = 0; bias < this.biases[l].length; bias++)
        this.biases[l][bias] = val === null ? Math.random() * 2 - 1 : val;
  }
}
