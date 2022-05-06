export function map(value, l, u, nL, nU) {
  return ((value - l) * (nU - nL)) / (u - l) + nL;
}

export function scaleImage(image, sf) {
  let out = [];

  const stride = 1 / sf;
  for (let row = 0; row < image.length; row += stride) {
    let outRow = [];
    for (let col = 0; col < image[0].length; col += stride) {
      outRow.push(image[~~row][~~col]);
    }
    out.push(outRow);
  }
  return out;
}

export function generateMatrix(width, height, initVal) {
  let out = [];

  for (let row = 0; row < height; row++) {
    let outRow = [];
    for (let col = 0; col < width; col++) outRow.push(initVal);
    out.push(outRow);
  }

  return out;
}

export function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export function dotProd(a, b) {
  if (a.length !== b.length) {
    console.log("function dotProd => Operand lengths do not match!");
    return null;
  }

  return a.reduce((total, aVal, i) => total + aVal * b[i], 0);
}

export function softmax(arr) {
  const total = arr.reduce((total, value) => total + value, 0);
  return arr.map((value) => Math.exp(value) / Math.exp(total));
}
