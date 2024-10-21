function loadImageToCanvas(imageFile, canvas) {
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.src = URL.createObjectURL(imageFile);

  img.onload = function () {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
  };
}

function applySobel(imageData) {
  const sobelX = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
  ];

  const sobelY = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
  ];

  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;
  const output = new Uint8ClampedArray(data.length);

  const convolve = (x, y, kernel) => {
    let r = 0,
      g = 0,
      b = 0;
    for (let ky = -1; ky <= 1; ky++) {
      for (let kx = -1; kx <= 1; kx++) {
        const pixelX = x + kx;
        const pixelY = y + ky;
        if (pixelX >= 0 && pixelX < width && pixelY >= 0 && pixelY < height) {
          const i = (pixelY * width + pixelX) * 4;
          const factor = kernel[ky + 1][kx + 1];
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
      const gx = convolve(x, y, sobelX);
      const gy = convolve(x, y, sobelY);
      output[i] = Math.sqrt(gx[0] ** 2 + gy[0] ** 2);
      output[i + 1] = Math.sqrt(gx[1] ** 2 + gy[1] ** 2);
      output[i + 2] = Math.sqrt(gx[2] ** 2 + gy[2] ** 2);
      output[i + 3] = 255;
    }
  }

  return new ImageData(output, width, height);
}

document.getElementById("sobelButton").addEventListener("click", function () {
  const canvas = document.getElementById("imageCanvas");
  const ctx = canvas.getContext("2d");
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const sobelData = applySobel(imageData);
  ctx.putImageData(sobelData, 0, 0);
});

document.getElementById("upload").addEventListener("change", function (event) {
  const file = event.target.files[0];
  if (file) {
    const canvas = document.getElementById("imageCanvas");
    loadImageToCanvas(file, canvas);
  }
});
