export function cost(O, Y) {
  if (O.length !== Y.length) {
    console.log("function cost => Operand lengths do not match!");
    return null;
  }

  return O.reduce((total, o, i) => total + Math.pow(o - Y[i], 2), 0) / O.length;
}
