import Network from "./network/Network.js";

import * as util from "./util.js";

import * as trainer from "./training/trainer.js";
import { cost } from "./training/trainingUtil.js";

// Inititalize network
const net = new Network(4, 5, 5, 4);
net.initBiases();
net.initWeights();

// Grab training data
true &&
  fetch("./trainingData/mirror.json")
    .then((res) => res.json())
    .then((trainingData) => {
      // Format data
      // console.log("Formatting data...");
      // trainingData = trainingData.map((sample) => {
      //   let formattedInput = [];
      //   sample.input.forEach((row) => row.forEach((value) => formattedInput.push((value + 1) / 2)));

      //   return {
      //     input: formattedInput,
      //     label: sample.label,
      //   };
      // });

      // Train network
      console.log("Training network...");
      // ==== Vars ====
      const learningRate = 1;
      const batchSize = 5;
      const itterationCount = 20000;
      // ==============
      for (let i = 0; i < itterationCount; i++) {
        i % (itterationCount / 10) === 0 && console.log("Training itteration:", i);
        const batch = trainer.generateBatch(trainingData, batchSize);
        const gradient = trainer.computeGradient(net, batch);
        trainer.applyGradient(net, gradient, learningRate);
      }

      // Test network
      for (let i = 0; i < 5; i++) {
        const sample = trainingData[~~(Math.random() * trainingData.length)];
        const netOut = net.calculate(sample.input);
        console.log("Inputting   :", sample.input.map((inp) => inp.toFixed(2))); // prettier-ignore
        console.log("Network out :", netOut.map((out) => out.toFixed(2))); // prettier-ignore
        console.log("Cost        :", cost(netOut, sample.label));
      }
    });
