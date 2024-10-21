function applyLaplacian(imageData) {
  const laplacianKernel = [
    [0, 0, -1, 0, 0],
    [0, -1, -2, -1, 0],
    [-1, -2, 16, -2, -1],
    [0, -1, -2, -1, 0],
    [0, 0, -1, 0, 0],
  ];

  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;
  const output = new Uint8ClampedArray(data.length);

  const convolve = (x, y, kernel) => {
    let r = 0,
      g = 0,
      b = 0;
    const offset = Math.floor(kernel.length / 2);
    for (let ky = -offset; ky <= offset; ky++) {
      for (let kx = -offset; kx <= offset; kx++) {
        const pixelX = x + kx;
        const pixelY = y + ky;
        if (pixelX >= 0 && pixelX < width && pixelY >= 0 && pixelY < height) {
          const i = (pixelY * width + pixelX) * 4;
          const factor = kernel[ky + offset][kx + offset];
          r += data[i] * factor;
          g += data[i + 1] * factor;
          b += data[i + 2] * factor;
        }
      }
    }
    return [Math.abs(r), Math.abs(g), Math.abs(b)];
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const [r, g, b] = convolve(x, y, laplacianKernel);
      output[i] = r;
      output[i + 1] = g;
      output[i + 2] = b;
      output[i + 3] = 255;
    }
  }

  return new ImageData(output, width, height);
}

document.getElementById("laplaceButton").addEventListener("click", function () {
  const canvas = document.getElementById("imageCanvas");
  const ctx = canvas.getContext("2d");
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const laplaceData = applyLaplacian(imageData);
  ctx.putImageData(laplaceData, 0, 0);
});
